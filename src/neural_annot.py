import anthropic
import json
from src.utils import TokenCorrelation, SubwordToken
import importlib
import os
os.environ.setdefault("ANTHROPIC_API_KEY", open("./data/anthropic_key.txt").read().strip())
from dataclasses import dataclass

# Map language name → tree-sitter package name (reused from symbolic_annot registry)
_TS_PACKAGE: dict[str, str] = {
    "Python":     "tree_sitter_python",
    "Rust":       "tree_sitter_rust",
    "Go":         "tree_sitter_go",
    "C":          "tree_sitter_c",
    "CPP":        "tree_sitter_cpp",
    "Java":       "tree_sitter_java",
    "JavaScript": "tree_sitter_javascript",
    "TypeScript": "tree_sitter_typescript",
    "C#":         "tree_sitter_c_sharp",
    "Ruby":       "tree_sitter_ruby",
    "Kotlin":     "tree_sitter_kotlin",
    "Swift":      "tree_sitter_swift",
    "Scala":      "tree_sitter_scala",
    "Haskell":    "tree_sitter_haskell",
    "PHP":        "tree_sitter_php",
    "Lua":        "tree_sitter_lua",
    "Shell":      "tree_sitter_bash",
    "R":          "tree_sitter_r",
    "Elixir":     "tree_sitter_elixir",
    "Erlang":     "tree_sitter_erlang",
}

_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}

# tree-sitter node types whose children form a (callee, args) call pattern
_CALL_NODE_TYPES = {
    "call_expression", "call", "function_call", "invocation_expression",
    "method_invocation", "method_call_expression",
}

# tree-sitter node types that represent a return statement
_RETURN_NODE_TYPES = {
    "return_statement", "return_expression",
}

# tree-sitter node types for typed parameters / local variable declarations
_TYPED_DECL_TYPES = {
    "local_variable_declaration", "variable_declaration",
    "typed_parameter", "parameter", "parameter_declaration",
    "formal_parameter",
    "let_declaration",   # Rust
    "property_declaration",  # Kotlin
}


@dataclass
class StructuralEdge:
    token_i_idx: int   # cue token index
    token_j_idx: int   # predicted token index
    reason: str        # 'bracket' | 'type' | 'return' | 'call' | 'loop'


class SyntacticCheckerTool:
    """
    Uses tree-sitter to extract directed structural edges:
        token[i]  causes / predicts  token[j]

    All edges have 100% structural confidence — no LLM involved.
    Feeds into NeuralAnnotator as grounding so the LLM can focus on
    semantic/dataflow edges only.
    """

    def get_edges(
        self,
        code: str,
        subwords: list[SubwordToken],
        language: str = "Python",
    ) -> list[StructuralEdge]:
        """
        Returns directed structural edges for `code`.
        Falls back gracefully to [] if tree-sitter is not installed.
        """
        pkg_name = _TS_PACKAGE.get(language)
        if pkg_name is None:
            return []
        try:
            from tree_sitter import Language, Parser
            pkg = importlib.import_module(pkg_name)
            lang = Language(pkg.language())
            parser = Parser(lang)
            tree = parser.parse(bytes(code, "utf-8"))
            root = tree.root_node
        except (ImportError, Exception):
            return []

        offset_to_idx = self._build_offset_map(subwords)
        edges: list[StructuralEdge] = []
        self._walk(root, code, offset_to_idx, subwords, edges)
        return edges

    # ── Build offset → token index map ───────────────────────────────────────

    @staticmethod
    def _build_offset_map(subwords: list[SubwordToken]) -> dict[int, int]:
        m: dict[int, int] = {}
        for i, sw in enumerate(subwords):
            for pos in range(sw.char_start, sw.char_end):  # char_end is already exclusive, no +1
                if pos not in m:
                    m[pos] = i
        return m

    # ── Resolve a tree-sitter byte span → (surface, token_idx) ───────────────

    @staticmethod
    def _resolve(
        start: int,
        end: int,
        code: str,
        offset_to_idx: dict[int, int],
        subwords: list[SubwordToken],
    ) -> int:
        """Returns token index, or -1 if not found."""
        for offset in range(start, end):
            if offset in offset_to_idx:
                return offset_to_idx[offset]
        return -1

    # ── Main entry: bracket pass first (global), then structural walk ─────────

    def _walk(self, root, code, offset_to_idx, subwords, edges):
        # Global passes (single traversal over the whole tree)
        self._bracket_edges(root, code, offset_to_idx, subwords, edges)
        self._defuse_edges(root, code, offset_to_idx, subwords, edges)
        # Per-node passes (visit each node once)
        self._walk_structural(root, code, offset_to_idx, subwords, edges)

    def _walk_structural(self, node, code, offset_to_idx, subwords, edges):
        if node.type in _CALL_NODE_TYPES:
            self._call_edges(node, code, offset_to_idx, subwords, edges)

        if node.type in _RETURN_NODE_TYPES:
            self._return_edges(node, code, offset_to_idx, subwords, edges)

        if node.type in _TYPED_DECL_TYPES:
            self._type_edges(node, code, offset_to_idx, subwords, edges)

        for child in node.children:
            self._walk_structural(child, code, offset_to_idx, subwords, edges)

    # ── Edge extractors ───────────────────────────────────────────────────────

    _CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}

    def _bracket_edges(self, root, code, offset_to_idx, subwords, edges):
        """
        Single global pass over all leaf nodes in document order.
        Uses a stack so arbitrarily nested brackets are matched correctly
        regardless of tree-sitter node depth.
        """
        stack: list[tuple[str, int]] = []  # (open_char, token_idx)

        def walk(node):
            if node.child_count == 0:  # leaf
                text = code[node.start_byte:node.end_byte]
                if text in _BRACKET_PAIRS:
                    i = self._resolve(node.start_byte, node.end_byte,
                                      code, offset_to_idx, subwords)
                    if i != -1:
                        stack.append((text, i))
                elif text in self._CLOSE_TO_OPEN:
                    target_open = self._CLOSE_TO_OPEN[text]
                    # pop back to find matching open
                    for k in range(len(stack) - 1, -1, -1):
                        if stack[k][0] == target_open:
                            _, open_idx = stack.pop(k)
                            j = self._resolve(node.start_byte, node.end_byte,
                                              code, offset_to_idx, subwords)
                            if j != -1 and open_idx != j:
                                edges.append(StructuralEdge(open_idx, j, "bracket"))
                            break
            for child in node.children:
                walk(child)

        walk(root)

    def _call_edges(self, node, code, offset_to_idx, subwords, edges):
        """callee token predicts each argument token."""
        # First child or field 'function'/'name' is typically the callee
        callee_node = (
            node.child_by_field_name("function")
            or node.child_by_field_name("name")
            or node.child_by_field_name("method")
            or (node.children[0] if node.children else None)
        )
        if callee_node is None:
            return
        i = self._resolve(callee_node.start_byte, callee_node.end_byte,
                          code, offset_to_idx, subwords)
        if i == -1:
            return

        # argument_list / arguments node
        args_node = (
            node.child_by_field_name("arguments")
            or node.child_by_field_name("argument_list")
        )
        if args_node is None:
            return
        for arg in args_node.children:
            if arg.type in (",", "(", ")", " "):
                continue
            j = self._resolve(arg.start_byte, arg.end_byte, code, offset_to_idx, subwords)
            if j != -1 and j != i:
                edges.append(StructuralEdge(i, j, "call"))

    def _return_edges(self, node, code, offset_to_idx, subwords, edges):
        """The `return` keyword predicts tokens in the returned expression."""
        # Find the `return` keyword child
        return_kw = next(
            (c for c in node.children
             if code[c.start_byte:c.end_byte] == "return"),
            None
        )
        if return_kw is None:
            return
        i = self._resolve(return_kw.start_byte, return_kw.end_byte,
                          code, offset_to_idx, subwords)
        if i == -1:
            return
        # All other children (the return value expression)
        for child in node.children:
            if child is return_kw or code[child.start_byte:child.end_byte] in (";", ""):
                continue
            j = self._resolve(child.start_byte, child.end_byte, code, offset_to_idx, subwords)
            if j != -1 and j != i:
                edges.append(StructuralEdge(i, j, "return"))

    def _defuse_edges(self, root, code, offset_to_idx, subwords, edges):
        """
        Global two-pass def-use analysis (replaces _loop_edges).
        Pass 1: collect all declaration sites → { surface → [token_idx, ...] }
        Pass 2: walk all leaf identifiers; emit decl_idx → use_idx for each match.
        Covers loop variables, let/var declarations, parameters, assignments.
        """
        DECL_PARENTS = _TYPED_DECL_TYPES | {
            "for_statement", "for_in_statement", "foreach_statement",
            "enhanced_for_statement", "for_expression",  # loop kinds
            "assignment", "assignment_expression", "short_var_declaration",
            "let_declaration", "variable_declarator", "local_variable_declaration",
        }
        IDENT_TYPES = {"identifier", "simple_identifier", "variable_name", "variable"}

        # ── Pass 1: find declared names ───────────────────────────────────────
        decl_map: dict[str, list[int]] = {}   # surface → [token_idx]

        def collect_decls(node):
            if node.type in DECL_PARENTS:
                name_node = next(
                    (c for c in node.children if c.type in IDENT_TYPES), None
                )
                if name_node:
                    surface = code[name_node.start_byte:name_node.end_byte]
                    idx = self._resolve(name_node.start_byte, name_node.end_byte,
                                        code, offset_to_idx, subwords)
                    if idx != -1:
                        decl_map.setdefault(surface, []).append(idx)
            for child in node.children:
                collect_decls(child)

        collect_decls(root)

        # ── Pass 2: walk all leaves; emit decl → use ─────────────────────────
        def collect_uses(node):
            if node.child_count == 0 and node.type in IDENT_TYPES:
                surface = code[node.start_byte:node.end_byte]
                if surface in decl_map:
                    j = self._resolve(node.start_byte, node.end_byte,
                                      code, offset_to_idx, subwords)
                    if j != -1:
                        for decl_idx in decl_map[surface]:
                            if decl_idx != j:
                                edges.append(StructuralEdge(decl_idx, j, "defuse"))
            for child in node.children:
                collect_uses(child)

        collect_uses(root)

    def _type_edges(self, node, code, offset_to_idx, subwords, edges):
        """
        In a typed declaration, type annotation tokens predict the variable name.
        Direction: type → varname  (seeing the type, you expect a name to follow).
        """
        # variable name is typically an identifier child
        name_node = next(
            (c for c in node.children
             if c.type in ("identifier", "simple_identifier", "variable_name")),
            None
        )
        type_node = (
            node.child_by_field_name("type")
            or node.child_by_field_name("type_annotation")
        )
        if name_node is None or type_node is None:
            return

        j = self._resolve(name_node.start_byte, name_node.end_byte, code, offset_to_idx, subwords)

        def collect_type_tokens(n):
            if n.type in ("identifier", "simple_identifier", "type_identifier",
                          "generic_type", "scoped_identifier"):
                yield n
            for c in n.children:
                yield from collect_type_tokens(c)

        for type_tok in collect_type_tokens(type_node):
            i = self._resolve(type_tok.start_byte, type_tok.end_byte, code, offset_to_idx, subwords)
            if i != -1 and j != -1 and i != j:
                edges.append(StructuralEdge(i, j, "type"))



class AnnotatorAgent:
    """
    NeuralAnnotator 重构为 tool-calling agent。
    工具列表：
      - get_structural_edges   : tree-sitter 结构边（原 SyntacticCheckerTool）
      - search_api_docs        : 查 API 文档（原 NeuralAnnotator 里的 web search）
      - emit_correlations      : 提交最终结果（终止信号）
    """

    def __init__(self, language: str = "Python", max_rounds: int = 6):
        self.language = language
        self.max_rounds = max_rounds
        self.client = anthropic.Anthropic()
        self._syntactic_tool = SyntacticCheckerTool()

    # ── 工具定义（告诉 LLM 有哪些工具）─────────────────────────────────────

    TOOLS = [
        {
            "name": "get_structural_edges",
            "description": (
                "Run tree-sitter on the code and return deterministic structural edges: "
                "bracket pairs, def-use, call arguments, return values, type annotations. "
                "Always call this first before semantic analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Programming language, e.g. 'Python'"},
                },
                "required": ["language"],
            },
        },
        {
            "name": "search_api_docs",
            "description": (
                "Look up documentation for a library function or type. "
                "Query must be specific: include the library name, function name, "
                "and what you want to know. "
                "Good: 'torch.nn.Linear parameters and return type' "
                "Bad: 'linear layer'"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "e.g. 'torch.nn.Linear arguments'}",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "emit_correlations",
            "description": (
                "Submit your semantic token correlations. "
                "Structural edges (bracket/defuse/call/return/type) are already recorded — "
                "only add edges beyond what the syntactic parser found. "
                "i and j are token indices — use exact indices to distinguish duplicate surface tokens. "
                "You MUST call this to finish."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "i":      {"type": "integer"},
                                "j":      {"type": "integer"},
                                "reason": {"type": "string"},
                            },
                            "required": ["i", "j", "reason"],
                        },
                        "description": "Semantic edges only (structural already confirmed).",
                    }
                },
                "required": ["pairs"],
            },
        },
    ]

    # ── 工具执行（你控制的部分）──────────────────────────────────────────────

    def _execute_tool(self, name: str, inputs: dict, code: str,
                      subwords: list[SubwordToken]) -> str:
        if name == "get_structural_edges":
            edges = self._syntactic_tool.get_edges(
                code, subwords, inputs["language"]
            )
            return json.dumps([
                {"i": e.token_i_idx, "j": e.token_j_idx, "reason": e.reason}
                for e in edges
            ])

        elif name == "search_api_docs":
            # 接你原来的 DuckDuckGo/OpenAI web search
            from src.web_search import search_docs  # 你已有的实现
            return search_docs(inputs["query"], language=self.language)

        elif name == "emit_correlations":
            # 终止信号，直接返回，agent 循环会检测到
            return "OK"

        return "Unknown tool"

    # ── Agent main loop ──────────────────────────────────────────────────────────

    def annotate(self, code: str, subwords: list[SubwordToken]) -> list[TokenCorrelation]:
        indexed = {i: sw.surface for i, sw in enumerate(subwords)}

        system = (
            f"You are a {self.language} code analysis assistant.\n"
            "Your task: identify DIRECTED PREDICTIVE token dependencies.\n"
            "A pair [i, j] means: token[i] CAUSES or PREDICTS token[j] — "
            "i.e. a programmer who has seen token[i] would EXPECT token[j] to appear at this position.\n\n"
            "Examples of valid [i, j] pairs:\n"
            "  - A variable's type token at i predicts the variable name token at j\n"
            "  - A function/method name at i predicts argument tokens and the closing ')' at j\n"
            "  - 'return' at i predicts the returned value token(s) at j\n"
            "  - A loop keyword ('for','while') at i predicts the loop variable token at j\n"
            "  - An opening bracket '(' at i predicts its matching ')' at j\n"
            "  - LHS of an assignment at i predicts the '=' at j and the RHS tokens at j\n"
            "  - A function definition name at i predicts its call-site name tokens at j\n"
            "  - An 'if' condition token at i predicts the 'else' keyword at j\n\n"
            "Directionality: i is the CUE, j is the CONSEQUENCE. i < j.\n\n"
            "Semantic edges to find:\n"
            "  - Variable declaration site predicts all use sites of that variable\n"
            "  - Function definition name predicts all call sites\n"
            "  - A specific value/literal at i predicts how it is used/transformed at j\n"
            "  - Conceptual coupling: tokens that belong to the same algorithm step\n\n"
            "Workflow:\n"
            "  1. Call get_structural_edges once to get confirmed structural edges.\n"
            "  2. Optionally call search_api_docs for unfamiliar API calls.\n"
            "  3. Call emit_correlations with your semantic edges — format: "
            "[{\"i\": int, \"j\": int, \"reason\": str}, ...]\n"
            "     Do NOT re-emit structural edges. You MUST call this to finish.\n\n"
            "Rules:\n"
            "  - Duplicate surface tokens are DISTINCT — use their exact indices.\n"
            "  - Ignore tokens inside docstrings and comments.\n"
            "  - Be exhaustive: each token[j] may have multiple predicting token[i]s."
            "  - Use one of these reason values: defuse, call, return, type, dataflow, semantic, api\n"
        )

        user = (
            f"Code:\n```{self.language}\n{code}\n```\n\n"
            "Start by calling get_structural_edges."
        )

        messages = [{"role": "user", "content": user}]
        final_pairs = []


        for _ in range(self.max_rounds):
            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=8092,
                system=system,
                tools=self.TOOLS,
                messages=messages,
                temperature=0,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                messages.append({"role": "user",
                                 "content": "You must call emit_correlations to submit your results. "
                                            "Do not end without calling it."})
                continue

            tool_results = []
            done = False
            for block in response.content:
                if block.type != "tool_use":
                    continue

                result = self._execute_tool(block.name, block.input, code, subwords)

                if block.name == "get_structural_edges":
                    structural_edges = json.loads(result)

                    # seed final_pairs with structural edges immediately
                    final_pairs = [
                        {"i": e["i"], "j": e["j"], "reason": e["reason"]}
                        for e in structural_edges
                    ]

                    by_reason = {}
                    for e in structural_edges:
                        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
                    summary = (
                            f"{len(structural_edges)} structural edges confirmed: "
                            + ", ".join(f"{v} {k}" for k, v in by_reason.items())
                            + ". Do NOT re-emit these — find semantic edges only."
                    )
                    result = (
                        f"Indexed tokens:\n{json.dumps(indexed)}\n\n"
                        f"Structural edges:\n{json.dumps(structural_edges)}\n\n"
                        f"# Summary: {summary}"
                    )

                elif block.name == "emit_correlations":
                    # append semantic edges on top of already-seeded structural ones
                    final_pairs += block.input.get("pairs", [])
                    done = True


                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
            if done:
                break
        else:
            import warnings
            warnings.warn(
                f"AnnotatorAgent exhausted max_rounds={self.max_rounds} without calling emit_correlations. "
                "Returning empty list.",
                RuntimeWarning,
            )

        _VALID_SUBTYPES = {"bracket", "defuse", "call", "return", "type", "dataflow", "semantic", "api", "syntactic"}

        return [
            TokenCorrelation(
                token_i=indexed.get(p["i"], ""),
                token_j=indexed.get(p["j"], ""),
                source="Neural",
                subtype=p.get("reason", "semantic") if p.get("reason") in _VALID_SUBTYPES else "semantic",
                token_i_idx=p["i"],
                token_j_idx=p["j"],
            )
            for p in final_pairs
            if p["i"] in indexed and p["j"] in indexed and p["i"] != p["j"]
        ]