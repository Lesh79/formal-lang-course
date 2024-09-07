from typing import Tuple

import cfpq_data
from networkx.drawing import nx_pydot
import networkx as nx


def load_graph(name: str) -> nx.MultiDiGraph:
    try:
        graph_path = cfpq_data.download(name)
        return cfpq_data.graph_from_csv(graph_path)
    except FileNotFoundError:
        raise FileNotFoundError


def graph_info(name: str):
    graph = load_graph(name)
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cfpq_data.get_sorted_labels(graph),
    )


def create_labeled_graph(
    count_of_nodes: int, count_of_edges: int, label: Tuple[str, str], path: str
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(
        n=count_of_nodes, m=count_of_edges, labels=label
    )
    save_graph_dot(graph, path)


def save_graph_dot(graph: nx.MultiDiGraph, path: str) -> None:
    grph = nx_pydot.to_pydot(graph)
    grph.write(path)
