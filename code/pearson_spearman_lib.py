import numpy as np
import pandas as pd
import scipy.stats as ss


def correlations_pearson(data):
    """
    Calculation of Pearson correlation between voxels.

    :param data: (array of voxels which value at time t=0 greater than eps, voxel's values at different time)
    :return: pandas DataFrame which is suitable for generating a graph by the igraph library.
    """
    correlations_matrix = np.corrcoef(data[1])
    size = correlations_matrix.shape[0]
    correlations_list = list(correlations_matrix[np.triu_indices(size, k=1)])
    source_list = []
    target_list = []
    tmp = size - 1
    for i in range(size - 1):
        source_list += [data[0][i] for j in range(tmp)]
        target_list += [data[0][j] for j in range(i + 1, size)]
        tmp -= 1
    correlations_df = pd.DataFrame({"source": source_list,
                                    "target": target_list,
                                    "correlation": correlations_list})
    return correlations_df


def correlations_spearman(data):
    """
    Calculation of Spearman correlation between voxels.

    :param data: (array of voxels which value at time t=0 greater than eps, voxel's values at different time)
    :return: pandas DataFrame which is suitable for generating a graph by the igraph library.
    """
    for i in range(len(data[0])):
        data[1][i] = ss.rankdata(data[1][i])
    correlations_matrix = np.corrcoef(data[1])
    size = correlations_matrix.shape[0]
    correlations_list = list(correlations_matrix[np.triu_indices(size, k=1)])
    source_list = []
    target_list = []
    tmp = size - 1
    for i in range(size - 1):
        source_list += [data[0][i] for j in range(tmp)]
        target_list += [data[0][j] for j in range(i + 1, size)]
        tmp -= 1
    correlations_df = pd.DataFrame({"source": source_list,
                                    "target": target_list,
                                    "correlation": correlations_list})
    return correlations_df


