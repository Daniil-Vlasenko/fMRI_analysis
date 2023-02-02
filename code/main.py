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


classifiers_folder = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/classifiers/SVC/max"
perception_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt"
imagery_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt"
shape = (20, 20, 16, 201)
edges_per_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/edges/perception/max.txt"
edges_ig_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/edges/imagery/max.txt"
graph_generation_lib.edges_calculation_2(classifiers_folder=classifiers_folder,
                                         perception_file=perception_file, imagery_file=imagery_file, shape=shape,
                                         edges_per_file=edges_per_file, edges_ig_file=edges_ig_file)

