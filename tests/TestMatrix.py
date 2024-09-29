from pyformlang.finite_automaton import Symbol

from project.task3 import AdjacencyMatrixFA
from project.task2 import regex_to_dfa


class TestMatrix:

    def test_matrix(self):
        dfa = regex_to_dfa("a.(a|b)*")
        adj = AdjacencyMatrixFA(dfa)
        assert adj.accepts([Symbol("a"), Symbol("b")])
        assert adj.accepts([Symbol("a"), Symbol("a")])
