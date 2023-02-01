from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle
from sklearn.svm import SVC
import numpy as np


def classifier_learning_1(perception_file, imagery_file, shape, regime):
    training_perception_file = open(perception_file, "r")
    training_imagery_file = open(imagery_file, "r")

    training_perception_lines = training_perception_file.readlines()
    training_imagery_lines = training_imagery_file.readlines()

    number_of_voxels = len(training_perception_lines)
    for voxel_1_id in range(number_of_voxels):
        unravel_id = np.unravel_index(voxel_1_id, shape[:3])
        voxels_ids_2 = [np.ravel_multi_index((unravel_id[0], unravel_id[1], unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2] - 1),  shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2]),  shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2] - 1), shape[:3])]
        for voxel_2_id in range(voxel_1_id + 1, number_of_voxels):
            print("voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)

            training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_1_id], dtype=float, sep=' ')
            training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_1_id], dtype=float, sep=' ')
            training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_2_id], dtype=float, sep=' ')
            training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_2_id], dtype=float, sep=' ')

            training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
            training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))

            training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
            training_y = [1 for i in range(len(training_perception_voxel1))] + \
                         [2 for j in range(len(training_imagery_voxel1))]

            # GPC = GaussianProcessClassifier(kernel=RBF(1), n_restarts_optimizer=0)
            # GPC.fit(training_X, training_y)
            # filename = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/GPC/mean/GPC_voxel_" + \
            #            str(voxel_1_id) + "_and_voxel_" + str(voxel_2_id) + ".sav"

            svc = SVC(kernel="rbf", C=25, probability=True)
            svc.fit(training_X, training_y)
            filename = "../correlations/training/dimensionality_reduction_1/10_10_10/synolitic_method_1/SVC/mean/SVC_voxel_" + \
                       str(voxel_1_id) + "_and_voxel_" + str(voxel_2_id) + ".sav"

            pickle.dump(svc, open(filename, 'wb'))
            # joblib.dump(svc, filename)