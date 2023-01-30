import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# mean
training_perception_file = open(
    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt", "r")
training_imagery_file = open(
    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt", "r")

training_perception_lines = training_perception_file.readlines()
training_imagery_lines = training_imagery_file.readlines()

number_of_voxels = len(training_perception_lines + training_imagery_lines)
for voxel_id_1 in range(number_of_voxels - 1):
    for voxel_id_2 in range(voxel_id_1 + 1, number_of_voxels):
        print("voxel_id_1:", voxel_id_1, "voxel_id_2:", voxel_id_2)

        training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_id_1], dtype=float, sep=' ')
        training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_id_1], dtype=float, sep=' ')
        training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_id_2], dtype=float, sep=' ')
        training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_id_2], dtype=float, sep=' ')

        training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
        training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))

        training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
        training_y = [1 for i in range(len(training_perception_voxel1))] + \
                     [2 for j in range(len(training_imagery_voxel1))]

        GPC = GaussianProcessClassifier(kernel=RBF(1), n_restarts_optimizer=0)
        GPC.fit(training_X, training_y)
