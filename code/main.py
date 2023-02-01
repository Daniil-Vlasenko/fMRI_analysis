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

cl.classifier_learning_1(cl.mean_10_10_10[0], cl.mean_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.median_10_10_10[0], cl.median_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.max_min_distance_10_10_10[0], cl.max_min_distance_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.quantiles_distance_10_10_10[0], cl.quantiles_distance_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.max_10_10_10[0], cl.max_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.min_10_10_10[0], cl.min_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.quantile_1_10_10_10[0], cl.quantile_1_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")
cl.classifier_learning_1(cl.quantile_2_10_10_10[0], cl.quantile_2_10_10_10[1], shape=cl.shape_10_10_10, regime="SVC")