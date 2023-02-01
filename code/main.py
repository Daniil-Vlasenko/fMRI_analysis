import pickle

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





import classifier_selection



# perception_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt"
# imagery_file = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt"
# shape = (20, 20, 16, 201)
# regime = "VSC"
#
# ### Функция
# training_perception_file = open(perception_file, "r")
# training_imagery_file = open(imagery_file, "r")
#
# training_perception_lines = training_perception_file.readlines()
# training_imagery_lines = training_imagery_file.readlines()
#
# number_of_voxels = len(training_perception_lines)
# for voxel_1_id in range(number_of_voxels):
#     unravel_id = np.unravel_index(voxel_1_id, shape[:3])
#     if unravel_id[0] == 0 or unravel_id[1] == 0 or unravel_id[2] == 0:
#         continue
#     voxels_ids_2 = [np.ravel_multi_index((unravel_id[0], unravel_id[1], unravel_id[2] - 1), shape[:3]),
#                     np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2]), shape[:3]),
#                     np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2] - 1), shape[:3]),
#                     np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2]), shape[:3]),
#                     np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2] - 1),  shape[:3]),
#                     np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2]),  shape[:3]),
#                     np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2] - 1), shape[:3])]
#     for voxel_2_id in voxels_ids_2:
#         print("voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)
#
#         training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_1_id], dtype=float, sep=' ')
#         training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_1_id], dtype=float, sep=' ')
#         training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_2_id], dtype=float, sep=' ')
#         training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_2_id], dtype=float, sep=' ')
#
#         training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
#         training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))
#
#         training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
#         training_y = [1 for i in range(len(training_perception_voxel1))] + \
#                      [2 for j in range(len(training_imagery_voxel1))]
#
#         # GPC = GaussianProcessClassifier(kernel=RBF(1), n_restarts_optimizer=0)
#         # GPC.fit(training_X, training_y)
#         # filename = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/GPC/mean/GPC_voxel_" + \
#         #            str(voxel_1_id) + "_and_voxel_" + str(voxel_2_id) + ".sav"
#
#         svc = SVC(kernel="rbf", C=25, probability=True)
#         svc.fit(training_X, training_y)
#         filename = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/SVC/mean/SVC_voxel_" + \
#                    str(voxel_1_id) + "_and_voxel_" + str(voxel_2_id) + ".sav"
#
#         pickle.dump(svc, open(filename, 'wb'))
#         # joblib.dump(svc, filename)
### Функция