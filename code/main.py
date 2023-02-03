import pickle

import joblib

import dimensionality_reduction_1_lib as dr
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
import pearson_spearman_lib as cor
import igraph as ig
from sklearn import svm
import preprocessed_data
import scalarization_lib
import processed_data_10_10_10
import processed_data_15_15_15
import processed_data_20_20_20
import processed_data_25_25_25
from pathlib import Path
import nilearn.image as image
import scalarization_lib as syn
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import classifier_learning_lib as cl
import graph_generation_lib
import scalars_data


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

# dataframe = pd.read_csv(edges_per_file)
# dataframe_0 = dataframe[["sours", "target", "0"]]
# g = ig.Graph.DataFrame(dataframe_0, directed=False)
# print(1)

# graph_generation_lib.graphs_generation(perception_file, imagery_file, edges_per_file, edges_ig_file, graph_per_folder, graph_im_folder)

# g = ig.read("../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/graphs/perception/run_0.gml")

cl.classifier_learning_1(scalars_data.max_10_10_10[0], scalars_data.max_10_10_10[1], shape, "SVC")