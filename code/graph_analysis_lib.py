import graphs_data_10_10_10 as gd10
import numpy as np


def get_ids_of_weak_voxels(graph, eps):
    return graph.vs(value_lt=eps)["flatidvoxel"]


def get_ids_of_edges_of_weak_voxels(graph, eps):
    voxels_ids = graph.vs(value_lt=eps)["flatidvoxel"]
    return graph.es(_source=voxels_ids)


def delete_edges_of_weak_voxels(graph, eps):
    edges_ids = get_ids_of_edges_of_weak_voxels(graph, eps)
    return graph.delete_edges(edges_ids)


def delete_wek_edges(graph, eps_1, eps_2):
    ideg = graph.es(value_ge=eps_1)
    idel = graph.es(value_le=eps_2)
    ide = np.intersect1d(idel, ideg)
    return graph.delete_edges(ide)


def number_of_edges(graph):
    return graph.ecount()


def number_of_vertices_le_degree(graph, degree):
    return len(graph.vs(_degree_le=degree))


def number_of_vertices_ge_degree(graph, degree):
    return len(graph.vs(_degree_ge=degree))


def number_of_vertices_eq_degree(graph, degree):
    return len(graph.vs(_degree_eq=degree))


def number_of_components(graph):
    return len(graph.connected_components())


def number_of_vertices_in_max_component(graph):
    components = graph.connected_components()
    numbers = [len(c) for c in components]
    return max(numbers)


def sum_of_edges(graph):  # !
    return np.sum(graph.es["value"])


def mean_of_edges(graph):   # !
    return np.mean(graph.es["value"])


def median_of_edges(graph):
    return np.median(graph.es["value"])


def sum_of_edges_gt_x(graph, x):
    return sum(value for value in graph.es["value"] if value > x)


def mean_of_edges_gt_x(graph, x):
    return np.mean([value for value in graph.es["value"] if value > x])


def median_of_edges_gt_x(graph, x):
    return np.median([value for value in graph.es["value"] if value > x])


def sum_of_edges_lt_x(graph, x):
    return sum(value for value in graph.es["value"] if value < x)


def mean_of_edges_lt_x(graph, x):
    return np.mean([value for value in graph.es["value"] if value < x])


def median_of_edges_lt_x(graph, x):
    return np.median([value for value in graph.es["value"] if value < x])


def get_vids_with_edges_weight_lt_q1(graph, prob):
    q_1 = np.quantile([value for value in graph.es["value"] if value < 0], prob)
    ideq_1 = graph.es(value_le=q_1)
    idvq_1 = [edge.target for edge in ideq_1] + [edge.source for edge in ideq_1]
    xidvq_1 = graph.vs(idvq_1)["xid"]
    yidvq_1 = graph.vs(idvq_1)["yid"]
    zidvq_1 = graph.vs(idvq_1)["zid"]
    return xidvq_1, yidvq_1, zidvq_1


def get_vids_with_edges_weight_gt_q2(graph, prob):
    q_2 = np.quantile([value for value in graph.es["value"] if value > 0], prob)
    ideq_2 = graph.es(value_ge=q_2)
    idvq_2 = [edge.target for edge in ideq_2] + [edge.source for edge in ideq_2]
    xidvq_2 = graph.vs(idvq_2)["xid"]
    yidvq_2 = graph.vs(idvq_2)["yid"]
    zidvq_2 = graph.vs(idvq_2)["zid"]
    return xidvq_2, yidvq_2, zidvq_2
