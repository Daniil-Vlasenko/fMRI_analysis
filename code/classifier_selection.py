import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# Create different classifiers.
classifiers = {
    "GPC RBF": GaussianProcessClassifier(kernel=RBF(1), n_restarts_optimizer=0, random_state=0),
    "SVC RBF C=25": SVC(kernel="rbf", C=25, probability=True, random_state=0),
    "L2 logistic C=25": LogisticRegression(penalty="l2", C=25, max_iter=10000),
}

training_accuracy = [0 for i in range(len(classifiers))]
test_accuracy = [0 for i in range(len(classifiers))]

# mean
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/mean.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/mean.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/mean.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/mean.txt", "r")
# median
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/median.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/median.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/median.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/median.txt", "r")
# max_min_distance
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/max_min_distance.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/max_min_distance.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/max_min_distance.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/max_min_distance.txt", "r")
# quantiles_distance
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/quantiles_distance.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/quantiles_distance.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/quantiles_distance.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/quantiles_distance.txt", "r")
# std
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/std.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/std.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/std.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/std.txt", "r")
# var
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/var.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/var.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/var.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/var.txt", "r")
# min
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/min.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/min.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/min.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/min.txt", "r")
# max
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/max.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/max.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/max.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/max.txt", "r")
# quantile_1
# training_perception_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/quantile_1.txt", "r")
# training_imagery_file = open(
#     "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/quantile_1.txt", "r")
# test_perception_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/quantile_1.txt", "r")
# test_imagery_file = open(
#     "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/quantile_1.txt", "r")
# quantile_2
training_perception_file = open(
    "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/quantile_2.txt", "r")
training_imagery_file = open(
    "../correlations/training/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/quantile_2.txt", "r")
test_perception_file = open(
    "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/perception/quantile_2.txt", "r")
test_imagery_file = open(
    "../correlations/test/dimensionality_reduction_1/20_20_20/synolitic_method_1/scalars/imagery/quantile_2.txt", "r")


training_perception_lines = training_perception_file.readlines()
training_imagery_lines = training_imagery_file.readlines()
test_perception_lines = test_perception_file.readlines()
test_imagery_lines = test_imagery_file.readlines()


# shape = (20,20,16,201)
shape = (11,11,9,201)
voxels1 = [np.ravel_multi_index((4,4,3), shape[:3]),
           np.ravel_multi_index((4,9,4), shape[:3]),
           np.ravel_multi_index((9,4,5), shape[:3]),
           np.ravel_multi_index((9,9,6), shape[:3]),
           np.ravel_multi_index((5,5,6), shape[:3]),
           np.ravel_multi_index((8,8,6), shape[:3]),
           np.ravel_multi_index((5,8,6), shape[:3]),
           np.ravel_multi_index((8,5,6), shape[:3])]
voxels2 = []
for voxel_id_1 in voxels1:
    unravel_id = np.unravel_index(voxels1[0], shape[:3])
    if unravel_id[0] == 0 or unravel_id[1] == 0 or unravel_id[2] == 0:
        print(0)
    voxels2 += [[np.ravel_multi_index((unravel_id[0], unravel_id[1], unravel_id[2] - 1), shape[:3]),
                    np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2]), shape[:3]),
                    np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2] - 1), shape[:3]),
                    np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2]), shape[:3]),
                    np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2] - 1),  shape[:3]),
                    np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2]),  shape[:3]),
                    np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2] - 1), shape[:3])]]

# for i in voxels1 + voxels2[0] + voxels2[1] + voxels2[2] + voxels2[3] + voxels2[4] + voxels2[5] + voxels2[6] + voxels2[7]:
#     print(i, training_perception_lines[i])

count = 0
for voxel_id_1 in voxels1:
    for voxel_id_2 in voxels1:
        if voxel_id_1 == voxel_id_2:
            continue
# for i, voxel_id_1 in enumerate(voxels1):
#     for voxel_id_2 in voxels2[i]:
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