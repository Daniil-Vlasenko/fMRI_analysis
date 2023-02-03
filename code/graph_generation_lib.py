import pickle
import joblib
import numpy as np
import pandas as pd
import igraph as ig
import scalars_data


def edges_calculation_1(voxel_1_value, voxel_2_value,  classifier):
    predict = classifier.predict_proba([voxel_1_value, voxel_2_value])
    return predict[1] - predict[0]


def edges_calculation_2(classifiers_folder, perception_file, imagery_file, shape, edges_per_file, edges_ig_file):
    """
    Calculate edge's weight for every two neighbour voxels of one scalar type for both perception and imagery regimes
    for every run of fMRI.

    :param classifiers_folder: string, path of folder where pre-calculated classifiers is.
    :param perception_file: string, path to perception scalars file.
    :param imagery_file: string, path to imagery scalars file.
    :param shape: (x, y, z, t), shape of the fMRI data.
    :param edges_per_file: string, path to new edge's weight dataframe for perception regime.
    :param edges_ig_file: string, path to new edge's weight dataframe for imagery regime.
    :return:
    """
    training_perception = np.loadtxt(perception_file)
    training_imagery = np.loadtxt(imagery_file)

    number_of_per_runs = len(training_perception[0])
    number_of_im_runs = len(training_imagery[0])
    number_of_voxels = len(training_perception)
    number_of_strings = 7 * (shape[0] * shape[1] * shape[2] - (shape[0] * shape[1] + shape[1] * shape[2] +
                             shape[0] * shape[2] - shape[0] - shape[1] - shape[2] + 1))

    edges_np_per = np.zeros((number_of_strings, number_of_per_runs + 2))
    edges_np_im = np.zeros((number_of_strings, number_of_im_runs + 2))

    count_of_rows = 0
    for voxel_1_id in range(number_of_voxels):
        unravel_id = np.unravel_index(voxel_1_id, shape[:3])
        if unravel_id[0] == 0 or unravel_id[1] == 0 or unravel_id[2] == 0:
            continue
        voxels_ids_2 = [np.ravel_multi_index((unravel_id[0], unravel_id[1], unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2] - 1), shape[:3])]
        for voxel_2_id in voxels_ids_2:
            print("voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)
            if "GPC" in classifiers_folder:
                classifier_file = classifiers_folder + "/GPC_voxel_" + str(voxel_1_id) + "_and_voxel_" + \
                                  str(voxel_2_id) + ".sav"
                classifier = joblib.load(classifier_file)
            else:
                classifier_file = classifiers_folder + "/SVC_voxel_" + str(voxel_1_id) + "_and_voxel_" + \
                                  str(voxel_2_id) + ".sav"
                classifier = pickle.load(open(classifier_file, 'rb'))

            edges_np_per[count_of_rows][0] = voxel_1_id
            edges_np_per[count_of_rows][1] = voxel_2_id
            edges_np_im[count_of_rows][0] = voxel_1_id
            edges_np_im[count_of_rows][1] = voxel_2_id


            tmp = pd.DataFrame(
                {'voxel1': training_perception[voxel_1_id], 'voxel2': training_perception[voxel_2_id]})
            tmp_per = classifier.predict_proba(tmp)
            edges_np_per[count_of_rows][2:] = tmp_per[:, 1] - tmp_per[:, 0]

            tmp = pd.DataFrame(
                {'voxel1': training_imagery[voxel_1_id], 'voxel2': training_imagery[voxel_2_id]})
            tmp_img = classifier.predict_proba(tmp)
            edges_np_im[count_of_rows][2:] = tmp_img[:, 1] - tmp_img[:, 0]

            count_of_rows += 1

    column_per_names = ["sours", "target"] + [str(i) for i in range(number_of_per_runs)]
    column_im_names = ["sours", "target"] + [str(i) for i in range(number_of_im_runs)]
    edges_df_per = pd.DataFrame(data=edges_np_per, columns=column_per_names)
    edges_df_img = pd.DataFrame(data=edges_np_im, columns=column_im_names)
    edges_df_per = edges_df_per.astype({"sours": "int", "target": "int"})
    edges_df_img = edges_df_img.astype({"sours": "int", "target": "int"})
    edges_df_per.to_csv(edges_per_file, index=False)
    edges_df_img.to_csv(edges_ig_file, index=False)


def properties_of_voxels(perception_file, imagery_file):
    training_perception = np.loadtxt(perception_file)
    training_imagery = np.loadtxt(imagery_file)
    # edges_df_per = pd.DataFrame(data=edges_np_per, columns=column_per_names)


def graphs_generation(perception_file, imagery_file, edges_per_file, edges_ig_file, graph_per_folder, graph_im_folder):
    df_per_edges = pd.read_csv(edges_per_file)
    df_im_edges = pd.read_csv(edges_ig_file)
    np_per_vertices = np.loadtxt(perception_file)
    np_im_vertices = np.loadtxt(imagery_file)

    number_of_per_runs = len(np_per_vertices[0])
    number_of_im_runs = len(np_im_vertices[0])
    number_of_voxels = len(np_per_vertices)
    id_vertices = [i for i in range(number_of_voxels)]

    for i in range(number_of_per_runs):
        dataframe_edges_per = df_per_edges[["sours", "target", str(i)]]
        dataframe_edges_per = dataframe_edges_per.rename(columns={"sours": "sours", "target": "target", str(i): "value"})
        dataframe_vertices_per = pd.DataFrame({'id': id_vertices, 'voxelid': id_vertices, "value": np_per_vertices[:, i]})
        g = ig.Graph.DataFrame(dataframe_edges_per, directed=False, vertices=dataframe_vertices_per)
        file_name = graph_per_folder + "/run_" + str(i) + ".gml"
        g.write(file_name, format="gml")
        print(i)

    for i in range(number_of_im_runs):
        dataframe_edges_im = df_im_edges[["sours", "target", str(i)]]
        dataframe_edges_im = dataframe_edges_im.rename(columns={"sours": "sours", "target": "target", str(i): "value"})
        dataframe_vertices_im = pd.DataFrame({'id': id_vertices, 'voxelid': id_vertices, "value": np_im_vertices[:, i]})
        g = ig.Graph.DataFrame(dataframe_edges_im, directed=False, vertices=dataframe_vertices_im)
        file_name = graph_im_folder + "/" + str(i) + ".gml"
        g.write(file_name, format="gml")
        print(-i)



classifiers_folder = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/classifiers/SVC/max"
perception_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt"
imagery_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt"
shape = (20, 20, 16, 201)
edges_per_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/edges/perception/max.txt"
edges_ig_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/edges/imagery/max.txt"
graph_per_folder = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/graphs/perception"
graph_im_folder =  "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/graphs/imagery"
# graph_generation_lib.edges_calculation_2(classifiers_folder=classifiers_folder,
#                                          perception_file=perception_file, imagery_file=imagery_file, shape=shape,
#                                          edges_per_file=edges_per_file, edges_ig_file=edges_ig_file)

# graph_generation_lib.graphs_generation(perception_file, imagery_file, edges_per_file, edges_ig_file, graph_per_folder, graph_im_folder)