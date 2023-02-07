import graphs_data_10_10_10 as gd10


def get_ids_of_empty_voxels(graph, eps):
    return graph.vs(value_lt=eps)["flatidvoxel"]


def get_ids_of_edges_of_empty_voxels(graph, eps):
    voxels_ids = graph.vs(value_lt=eps)["flatidvoxel"]
    return graph.es(_source=voxels_ids)


def delete_edges_of_empty_voxels(graph, eps):
    edges_ids = get_ids_of_edges_of_empty_voxels(graph, eps)
    return graph.delete_edges(edges_ids)


