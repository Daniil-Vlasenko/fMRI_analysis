import joblib
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def get_set_of_neighbors(shape):
    set_ = set()
    for x in range(1, shape[0] - 1):
        for y in range(1, shape[1] - 1):
            for z in range(1, shape[2] - 1):
                voxel_id = np.ravel_multi_index((x, y, z), shape[:3])
                neighbors = [tuple(sorted([np.ravel_multi_index((x - 1, y - 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y + 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y - 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y + 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y - 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y + 1, z - 1), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x - 1, y - 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y + 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y - 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y + 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y - 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y + 1, z + 1), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x - 1, y - 1, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y + 1, z), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x + 1, y - 1, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y + 1, z), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x, y - 1, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y + 1, z), shape[:3]), voxel_id]))]
                set_.update(neighbors)
    return set_


def classifier_learning_1(perception_file, imagery_file, shape, regime):
    training_perception_file = open(perception_file, "r")
    training_imagery_file = open(imagery_file, "r")

    training_perception_lines = training_perception_file.readlines()
    training_imagery_lines = training_imagery_file.readlines()

    count = 0
    for x in range(1, shape[0] - 1, 2):
        for y in range(1, shape[1] - 1, 2):
            for z in range(1, shape[2] - 1, 2):
                voxel_1_id = np.ravel_multi_index((x, y, z), shape[:3])
                voxels_ids_2 = [np.ravel_multi_index((x - 1, y - 1, z - 1), shape[:3]),
                                np.ravel_multi_index((x - 1, y, z - 1), shape[:3]),
                                np.ravel_multi_index((x - 1, y + 1, z - 1), shape[:3]),
                                np.ravel_multi_index((x + 1, y - 1, z - 1), shape[:3]),
                                np.ravel_multi_index((x + 1, y, z - 1), shape[:3]),
                                np.ravel_multi_index((x + 1, y + 1, z - 1), shape[:3]),
                                np.ravel_multi_index((x, y - 1, z - 1), shape[:3]),
                                np.ravel_multi_index((x, y, z - 1), shape[:3]),
                                np.ravel_multi_index((x, y + 1, z - 1), shape[:3]),

                                np.ravel_multi_index((x - 1, y - 1, z + 1), shape[:3]),
                                np.ravel_multi_index((x - 1, y, z + 1), shape[:3]),
                                np.ravel_multi_index((x - 1, y + 1, z + 1), shape[:3]),
                                np.ravel_multi_index((x + 1, y - 1, z + 1), shape[:3]),
                                np.ravel_multi_index((x + 1, y, z + 1), shape[:3]),
                                np.ravel_multi_index((x + 1, y + 1, z + 1), shape[:3]),
                                np.ravel_multi_index((x, y - 1, z + 1), shape[:3]),
                                np.ravel_multi_index((x, y, z + 1), shape[:3]),
                                np.ravel_multi_index((x, y + 1, z + 1), shape[:3]),

                                np.ravel_multi_index((x - 1, y - 1, z), shape[:3]),
                                np.ravel_multi_index((x - 1, y, z), shape[:3]),
                                np.ravel_multi_index((x - 1, y + 1, z), shape[:3]),

                                np.ravel_multi_index((x + 1, y - 1, z), shape[:3]),
                                np.ravel_multi_index((x + 1, y, z), shape[:3]),
                                np.ravel_multi_index((x + 1, y + 1, z), shape[:3]),

                                np.ravel_multi_index((x, y - 1, z), shape[:3]),
                                np.ravel_multi_index((x, y + 1, z), shape[:3])]
                for voxel_2_id in voxels_ids_2:
                    count += 1
                    print("count:", count, "voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)

                    training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_1_id], dtype=float, sep=' ')
                    training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_1_id], dtype=float, sep=' ')
                    training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_2_id], dtype=float, sep=' ')
                    training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_2_id], dtype=float, sep=' ')

                    training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
                    training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))

                    training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
                    training_y = [1 for i in range(len(training_perception_voxel1))] + \
                                 [2 for j in range(len(training_imagery_voxel1))]

                    path = perception_file.split(sep="/")
                    if regime == "GPC":
                        gpc = GaussianProcessClassifier(kernel=RBF(1))
                        gpc.fit(training_X, training_y)
                        filename = path[0] + "/" + path[1] + "/" + path[2] + "/" + path[3] + "/" + path[4] + "/" + path[5] + \
                                   "/classifiers/GPC/" + path[-1][:-4] + "/GPC_voxel_" + str(voxel_1_id) + "_and_voxel_" + str(
                            voxel_2_id) + ".sav"
                        joblib.dump(gpc, filename)
                    else:
                        svc = SVC(kernel="rbf", C=25, probability=True)
                        svc.fit(training_X, training_y)
                        filename = path[0] + "/" + path[1] + "/" + path[2] + "/" + path[3] + "/" + path[4] + "/" + path[5] + \
                                   "/classifiers/SVC/" + path[-1][:-4] + "/SVC_voxel_" + str(voxel_1_id) + "_and_voxel_" + str(
                            voxel_2_id) + ".sav"
                        pickle.dump(svc, open(filename, 'wb'))