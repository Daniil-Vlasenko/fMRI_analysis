import dimensionality_reduction_1 as dr
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
import pearson_spearman as cor
import igraph as ig
from sklearn import svm
import preprocessed_data
import synolitic
import processed_data
from pathlib import Path
import nilearn.image as image
import synolitic as syn

# print(len(processed_data.imagery))
# print(len(processed_data.perception))
# print(len(processed_data.imagery_test))
# print(len(processed_data.perception_test))
# print(len(processed_data.imagery_training))
# print(len(processed_data.perception_training))


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.gaussian_process.kernels as kernels

C = 25
# Create different classifiers.
classifiers = {
    "GPC RBF": GaussianProcessClassifier(kernel=kernels.RBF(), random_state=0),
    "SVC C=25": SVC(kernel="rbf", C=C, probability=True, random_state=0),
    "L2 logistic": LogisticRegression(penalty="l2", C=C, max_iter=10000),
}

training_accuracy = [0 for i in range(len(classifiers))]
test_accuracy = [0 for i in range(len(classifiers))]
training_perception_file = open("../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt", "r")
training_imagery_file = open("../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt", "r")
test_perception_file = open("../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt", "r")
test_imagery_file = open("../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt", "r")

training_perception_lines = training_perception_file.readlines()
training_imagery_lines = training_imagery_file.readlines()
test_perception_lines = test_perception_file.readlines()
test_imagery_lines = test_imagery_file.readlines()

# voxels1 = [1000, 2000, 3000, 4000, 5000, 6000]
voxels1 = [1000, 2000, 3000, 4000]
# voxels1 = [1000]
# voxels2 = [500, 1500, 2500, 3500, 4500, 5500]
voxels2 = [1001, 999, 2001, 1999, 3001, 2999, 4001, 3999]
# voxels2 = [2500]

count = 0
for voxel_id_1 in voxels1:
    for voxel_id_2 in voxels2:
        count += 1
        print("voxel_id_1:", voxel_id_1, "voxel_id_2:", voxel_id_2)

        training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_id_1], dtype=float, sep=' ')
        training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_id_2], dtype=float, sep=' ')
        training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_id_1], dtype=float, sep=' ')
        training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_id_2], dtype=float, sep=' ')

        test_perception_voxel1 = np.fromstring(test_perception_lines[voxel_id_1], dtype=float, sep=' ')
        test_perception_voxel2 = np.fromstring(test_perception_lines[voxel_id_2], dtype=float, sep=' ')
        test_imagery_voxel1 = np.fromstring(test_imagery_lines[voxel_id_1], dtype=float, sep=' ')
        test_imagery_voxel2 = np.fromstring(test_imagery_lines[voxel_id_2], dtype=float, sep=' ')

        training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
        training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))
        test_voxel1 = np.concatenate((test_perception_voxel1, test_imagery_voxel1))
        test_voxel2 = np.concatenate((test_perception_voxel2, test_imagery_voxel2))

        training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
        training_y = [1 for i in range(len(training_perception_voxel1))] + [2 for j in range(len(training_imagery_voxel1))]

        test_X = pd.DataFrame({'voxel1': test_voxel1, 'voxel2': test_voxel2})
        test_y = [1 for k in range(len(test_perception_voxel1))] + [2 for l in range(len(test_imagery_voxel1))]

        training_y_pred = []
        test_y_pred = []
        for index, (name, classifier) in enumerate(classifiers.items()):
            # print(name)
            classifier.fit(training_X, training_y)

            training_y_pred = classifier.predict(training_X)
            training_accuracy[index] += accuracy_score(training_y, training_y_pred) * 100

            test_y_pred = classifier.predict(test_X)
            test_accuracy[index]  += accuracy_score(test_y, test_y_pred) * 100

for i in range(len(classifiers)):
    training_accuracy[i] /= count
    test_accuracy[i] /= count
print(training_accuracy)
print(test_accuracy)