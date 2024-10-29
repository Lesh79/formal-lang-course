from typing import Any, Dict, Set, Tuple
import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_matrix

from project.cfg import cfg_to_weak_normal_form


def build_index_maps(graph: nx.DiGraph) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}
    index_to_node = {idx: node for node, idx in node_to_index.items()}
    return node_to_index, index_to_node


def initialize_matrices(
    graph: nx.DiGraph, cfg: CFG, node_to_index: Dict[Any, int]
) -> Dict[Variable, csr_matrix]:
    var_matrices = {}
    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label:
            for prod in cfg.productions:
                if len(prod.body) == 1 and isinstance(prod.body[0], Terminal):
                    if prod.body[0].value == label:
                        head = prod.head
                        if head not in var_matrices:
                            var_matrices[head] = csr_matrix(
                                (len(node_to_index), len(node_to_index)), dtype=bool
                            )
                        var_matrices[head][node_to_index[u], node_to_index[v]] = True
    return var_matrices


def add_nullable_diagonals(
    graph: nx.DiGraph,
    cfg: CFG,
    node_to_index: Dict[Any, int],
    var_matrices: Dict[Variable, csr_matrix],
):
    for var in cfg.get_nullable_symbols():
        variable = Variable(var.value)
        if variable not in var_matrices:
            var_matrices[variable] = csr_matrix(
                (len(node_to_index), len(node_to_index)), dtype=bool
            )
        for node in graph.nodes:
            var_matrices[variable][node_to_index[node], node_to_index[node]] = True


def apply_productions(cfg: CFG, var_matrices: Dict[Variable, csr_matrix]) -> bool:
    added = False
    for prod in cfg.productions:
        if len(prod.body) == 2:
            left_var, right_var = (
                Variable(prod.body[0].value),
                Variable(prod.body[1].value),
            )
            if left_var in var_matrices and right_var in var_matrices:
                head = prod.head
                if head not in var_matrices:
                    var_matrices[head] = csr_matrix(
                        var_matrices[left_var].shape, dtype=bool
                    )

                new_entries = (var_matrices[left_var] @ var_matrices[right_var]).tocoo()
                for u, v in zip(new_entries.row, new_entries.col):
                    if not var_matrices[head][u, v]:
                        var_matrices[head][u, v] = True
                        added = True
    return added


def extract_results(
    var_matrices: Dict[Variable, csr_matrix],
    index_to_node: Dict[int, Any],
    start_nodes: Set[int],
    final_nodes: Set[int],
    start_symbol: Variable,
) -> Set[Tuple[int, int]]:
    result_pairs = set()
    if start_symbol in var_matrices:
        for u_idx, v_idx in zip(*var_matrices[start_symbol].tocoo().nonzero()):
            u, v = index_to_node[u_idx], index_to_node[v_idx]
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result_pairs.add((u, v))
    return result_pairs


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    weak_normal_form = cfg_to_weak_normal_form(cfg)
    node_to_index, index_to_node = build_index_maps(graph)
    var_matrices = initialize_matrices(graph, weak_normal_form, node_to_index)
    add_nullable_diagonals(graph, weak_normal_form, node_to_index, var_matrices)

    while apply_productions(weak_normal_form, var_matrices):
        pass

    return extract_results(
        var_matrices,
        index_to_node,
        start_nodes,
        final_nodes,
        weak_normal_form.start_symbol,
    )
