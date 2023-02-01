import pickle

import joblib
import numpy as np
import pandas as pd
import igraph as ig
import scalars_data

def edges_calculation_1(voxel_1_value, voxel_2_value,  classifier):
    predict = classifier.predict_proba([voxel_1_value, voxel_2_value])
    return predict[1] - predict[0]


def edges_calculation_2(classifiers_folder, regime, perception_file, imagery_file, shape, edges_file):
    training_perception_file = open(perception_file, "r")
    training_imagery_file = open(imagery_file, "r")

    training_perception_lines = training_perception_file.readlines()
    training_imagery_lines = training_imagery_file.readlines()

    number_of_voxels = len(training_perception_lines)
    for voxel_1_id in range(number_of_voxels):
        unravel_id = np.unravel_index(voxel_1_id, shape[:3])
        if unravel_id[0] == 0 or unravel_id[1] == 0 or unravel_id[2] == 0:
            continue
        voxels_ids_2 = [np.ravel_multi_index((unravel_id[0], unravel_id[1], unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0], unravel_id[1] - 1, unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1], unravel_id[2] - 1), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2]), shape[:3]),
                        np.ravel_multi_index((unravel_id[0] - 1, unravel_id[1] - 1, unravel_id[2] - 1), shape[:3])]
        for voxel_2_id in voxels_ids_2:
            print("voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)
            classifier = 0
            if regime == "GPC":
                classifier_file = classifiers_folder + "/GPC/GPC_voxel_" + str(voxel_1_id) + "_and_voxel_" + \
                                  str(voxel_2_id) + ".sav"
                classifier = joblib.load(classifier_file)
            else:
                classifier_file = classifiers_folder + "/SVC/SVC_voxel_" + str(voxel_1_id) + "_and_voxel_" + \
                                  str(voxel_2_id) + ".sav"
                classifier = loaded_model = pickle.load(open(classifier_file, 'rb'))


