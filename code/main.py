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
from sklearn.gaussian_process.kernels import RBF


training_accuracy = [0, 0, 0, 0, 0]
test_accuracy = [0, 0, 0, 0, 0]
with open("../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt") as training_perception_file, open(
        "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt") as training_imagery_file, open(
        "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/perception/mean.txt") as test_perception_file, open(
        "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt") as test_imagery_file:
    training_perception_lines = training_perception_file.readlines()
    training_imagery_lines = training_imagery_file.readlines()
    test_perception_lines = test_perception_file.readlines()
    test_imagery_lines = test_imagery_file.readlines()

    # for voxel_id_1 in range(len(training_perception_lines) - 1):
    for voxel_id_1 in range(1000, 1001):
        print("voxel_id_1:", voxel_id_1)
        # for voxel_id_2 in range(voxel_id_1 + 1, len(training_perception_lines)):
        for voxel_id_2 in (1100, 2100, 3100, 4100, 5100, 100, 500, 1500, 2000, 2500):
            print("voxel_id_2:", voxel_id_2)
            training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_id_1], dtype=float, sep=' ')
            training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_id_2], dtype=float, sep=' ')
            training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_id_1], dtype=float, sep=' ')
            training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_id_2], dtype=float, sep=' ')

            test_perception_voxel1 = np.fromstring(test_perception_lines[voxel_id_1], dtype=float, sep=' ')
            test_perception_voxel2 = np.fromstring(test_perception_lines[voxel_id_2], dtype=float, sep=' ')
            test_imagery_voxel1 = np.fromstring(test_imagery_lines[voxel_id_1], dtype=float, sep=' ')
            test_imagery_voxel2 = np.fromstring(test_imagery_lines[voxel_id_2], dtype=float, sep=' ')

            # fig, ax = plt.subplots()
            # ax.scatter(training_perception_voxel1, training_perception_voxel2, label="p")
            # ax.scatter(training_imagery_voxel1, training_imagery_voxel2, label="i")
            # plt.show()

            training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
            training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))
            test_voxel1 = np.concatenate((test_perception_voxel1, test_imagery_voxel1))
            test_voxel2 = np.concatenate((test_perception_voxel2, test_imagery_voxel2))

            training_X = pd.DataFrame(
                {'voxel1': training_voxel1,
                 'voxel2': training_voxel2
                 })
            training_y = [1 for i in range(len(training_perception_voxel1))] + [2 for j in range(len(training_imagery_voxel1))]
            test_X = pd.DataFrame(
                {'voxel1': test_voxel1,
                 'voxel2': test_voxel2
                 })
            test_y = [1 for k in range(len(test_perception_voxel1))] + [2 for l in range(len(test_imagery_voxel1))]

            C = 10
            kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

            # Create different classifiers.
            classifiers = {
                "L1 logistic": LogisticRegression(
                    C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
                ),
                "L2 logistic (Multinomial)": LogisticRegression(
                    C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
                ),
                "L2 logistic (OvR)": LogisticRegression(
                    C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
                ),
                "Linear SVC": SVC(kernel="rbf", C=C, probability=True, random_state=0),
                "GPC": GaussianProcessClassifier(kernel),
            }

            training_y_pred = []
            test_y_pred = []
            for index, (name, classifier) in enumerate(classifiers.items()):
                classifier.fit(training_X, training_y)

                training_y_pred = classifier.predict(training_X)
                training_accuracy[index] += accuracy_score(training_y, training_y_pred) * 100

                test_y_pred = classifier.predict(test_X)
                test_accuracy[index]  += accuracy_score(test_y, test_y_pred) * 100

for i in range(5):
    training_accuracy[i] /= 10
    test_accuracy[i] /= 10
print(training_accuracy)
print(test_accuracy)