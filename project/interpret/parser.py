from antlr4 import (
    InputStream,
    CommonTokenStream,
    ParserRuleContext,
    TerminalNode,
)

from project.interpret.GraphQueryLanguageLexer import GraphQueryLanguageLexer
from project.interpret.GraphQueryLanguageParser import GraphQueryLanguageParser


def get_parser(input_program: str) -> GraphQueryLanguageParser:
    input_stream = InputStream(input_program)
    lexer = GraphQueryLanguageLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    return GraphQueryLanguageParser(token_stream)


def parse(input_program: str) -> ParserRuleContext:
    parser = get_parser(input_program)
    return parser.prog()


def accepts(input_program: str) -> bool:
    parser = get_parser(input_program)
    parser.removeErrorListeners()
    parser.prog()
    return parser.getNumberOfSyntaxErrors() == 0


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    try:
        parser = get_parser(program)
        syntax_tree = parser.prog()
        success = parser.getNumberOfSyntaxErrors() == 0
        return syntax_tree, success
    except Exception:
        return None, False


def tree_to_program(tree: ParserRuleContext) -> str:
    if isinstance(tree, TerminalNode):
        return tree.getText()

    program_parts = []
    for child in tree.getChildren():
        child_text = tree_to_program(child)
        if program_parts and child_text not in ("", "(", ")", "{", "}", ",", ";"):
            program_parts.append(" ")
        program_parts.append(child_text)

    return "".join(program_parts)


def nodes_count(tree: ParserRuleContext) -> int:
    if isinstance(tree, TerminalNode):
        return 1

    return 1 + sum(nodes_count(child) for child in tree.getChildren())
