from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.fa_module import regex_to_dfa, graph_to_nfa
from project.graph_module import load_graph


class TestDFA:
    def test_regex_to_dfa(self):
        regex = "xy* (x | y*)"
        dfa = regex_to_dfa(regex)

        excepted = NondeterministicFiniteAutomaton()
        excepted.add_start_state(0)
        excepted.add_final_state(0)
        excepted.add_final_state(1)
        excepted.add_final_state(2)
        excepted.add_transitions([(0, "xy", 0), (0, "y", 1), (1, "y", 1), (0, "x", 2)])
        assert excepted.is_equivalent_to(dfa)

    def test_build_dfa_from_regex(self):
        regex = "xy* (x | y*) | ab (x | y*) | (x | a*) (x | y*)"
        dfa = regex_to_dfa(regex)
        assert dfa.is_deterministic()
        dfa_minimize = dfa.minimize()
        assert dfa.is_equivalent_to(dfa_minimize)

    def test_build_from_loaded_graph(self):
        graph = load_graph("wc")
        nfa = graph_to_nfa(graph)

        excepted_graph_info = 332

        assert len(nfa.states) == excepted_graph_info
        assert len(nfa.final_states) == excepted_graph_info
        assert len(nfa.start_states) == excepted_graph_info
