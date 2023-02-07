import classifier_learning_lib
import dimensionality_reduction_1_lib as dr
import pearson_spearman_lib as cor
import preprocessed_data
import scalarization_lib
import processed_data_10_10_10
import processed_data_15_15_15
import processed_data_20_20_20
import processed_data_25_25_25
import scalarization_lib as syn
import classifier_learning_lib as cl
import graph_generation_lib
import scalars_data_10_10_10 as scalars
import edges_data_10_10_10 as edges
import graphs_data_10_10_10 as graphs
import classifiers_data_10_10_10 as classifiers


shape = (20, 20, 16, 201)

# for i in range(8):
#     cl.classifier_learning_1_SVC(perception_file=scalars.scalars[i][0], imagery_file=scalars.scalars[i][1],
#                                  shape=shape, classifiers_folder=classifiers.classifiers[i])
# print("classifier_learning_1_SVC end")
# for i in range(8):
#     graph_generation_lib.edges_calculation_1(classifiers_folder=classifiers.classifiers[i],
#                                              perception_file=scalars.scalars[i][0], imagery_file=scalars.scalars[i][1],
#                                              shape=shape, edges_per_file=edges.edges[i][0], edges_ig_file=edges.edges[i][1])
# print("edges_calculation_1 end")
for i in range(8):
    graph_generation_lib.graphs_generation(perception_file=scalars.scalars[i][0], imagery_file=scalars.scalars[i][1],
                                           shape=shape, edges_per_file=edges.edges[i][0], edges_ig_file=edges.edges[i][1],
                                           graph_per_folder=graphs.graphs[i][0], graph_im_folder=graphs.graphs[i][1])
