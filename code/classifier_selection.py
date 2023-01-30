import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

C = 25
# Create different classifiers.
classifiers = {
    "GPC RBF": GaussianProcessClassifier(kernel=RBF(1), n_restarts_optimizer=0, random_state=0),
    "SVC RBF C=25": SVC(kernel="rbf", C=C, probability=True, random_state=0),
    "L2 logistic C=25": LogisticRegression(penalty="l2", C=C, max_iter=10000),
}

training_accuracy = [0 for i in range(len(classifiers))]
test_accuracy = [0 for i in range(len(classifiers))]

# mean
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt", "r")
# median
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/median.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/median.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/median.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/median.txt", "r")
# min_max_dist
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min_max_dist.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min_max_dist.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min_max_dist.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min_max_dist.txt", "r")
# quant_dist
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quant_dist.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quant_dist.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quant_dist.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quant_dist.txt", "r")
# std
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/std.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/std.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/std.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/std.txt", "r")
# var
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/var.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/var.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/var.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/var.txt", "r")
# min
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/min.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/min.txt", "r")
# max
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/max.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/max.txt", "r")
# quantile_1
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_1.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_1.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_1.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_1.txt", "r")
# quantile_2
training_perception_file = open(
    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_2.txt", "r")
training_imagery_file = open(
    "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_2.txt", "r")
test_perception_file = open(
    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/quantile_2.txt", "r")
test_imagery_file = open(
    "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/quantile_2.txt", "r")


training_perception_lines = training_perception_file.readlines()
training_imagery_lines = training_imagery_file.readlines()
test_perception_lines = test_perception_file.readlines()
test_imagery_lines = test_imagery_file.readlines()

# not neighbors:
# voxels1 = [500, 2500, 3500, 4500, 5500]
# voxels2 = [1000, 1500, 3000, 5000]

# neighbors:
voxels1 = [1000, 3000, 5000]
voxels2 = [[997, 998, 999, 1001, 1002, 1003], [2997, 2998, 2999, 3001, 3002, 3003], [4997, 4998, 4999, 5001, 5002, 5003]]

# for i in voxels1 + voxels2:
#     print(i, training_perception_lines[i])

count = 0
# for voxel_id_1 in voxels1:
#     for voxel_id_2 in voxels2:
for i, voxel_id_1 in enumerate(voxels1):
    for voxel_id_2 in voxels2[i]:
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
        training_y = [1 for i in range(len(training_perception_voxel1))] + [2 for j in
                                                                            range(len(training_imagery_voxel1))]

        test_X = pd.DataFrame({'voxel1': test_voxel1, 'voxel2': test_voxel2})
        test_y = [1 for k in range(len(test_perception_voxel1))] + [2 for l in range(len(test_imagery_voxel1))]

        training_y_pred = []
        test_y_pred = []
        for j, (name, classifier) in enumerate(classifiers.items()):
            print(name)
            classifier.fit(training_X, training_y)

            # training_y_pred = classifier.predict(training_X)
            # training_accuracy[j] += accuracy_score(training_y, training_y_pred) * 100

            test_y_pred = classifier.predict(test_X)
            test_accuracy[j] += accuracy_score(test_y, test_y_pred) * 100

for i in range(len(classifiers)):
    test_accuracy[i] /= count
    # training_accuracy[i] /= count

for j, name in enumerate(classifiers.keys()):
    print(name + ": " + str(test_accuracy[j]) + " %", end="; ")