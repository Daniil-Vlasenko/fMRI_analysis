import nibabel as nib
import numpy as np
import pandas as pd
import dimensionality_reduction as dr


def scalarization_1(data, regime, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :param regime: numeric, regime of function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance,
    4 - difference between the maximum and minimum, 5 - difference between quantiles.
    :param q_1: numeric, level of quantile 1.
    :param q_2:numeric, level of quantile 2.
    :return: ndarray, array of scalars.
    """
    if regime == 0:
        return mean(data)
    elif regime == 1:
        return median(data)
    elif regime == 2:
        return sd(data)
    elif regime == 3:
        return var(data)
    elif regime == 4:
        return max_min_distance(data)
    elif regime == 5:
        return quantiles_distance(data, q_1, q_2)
    else:
        raise Exception("Invalid regime.")


def scalarization_2(file_name, regime, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value.

    :param file_name: string, name of input file.
    :param regime: numeric, regime of function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance,
    4 - difference between the maximum and minimum, 5 - difference between quantiles.
    :param q_1: numeric, level of quantile 1.
    :param q_2:numeric, level of quantile 2.
    :return: ndarray, array of scalars.
    """
    img = nib.load(file_name)
    data = img.get_fdata()
    data = dr._4D_to_2D(data)
    return scalarization_1(data, regime, q_1, q_2)


def scalarization_3(file_names, file_name, regime, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value and save them to file.

    :param file_names: list of strings, names of input files.
    :param file_name: string, name of output file.
    :param regime: numeric, regime of function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance,
    4 - difference between the maximum and minimum, 5 - difference between quantiles.
    :param q_1: numeric, level of quantile 1.
    :param q_2:numeric, level of quantile 2.
    :return: ndarray, array of scalars.
    """
    data = scalarization_2(file_names[0], regime)
    size = len(file_names)
    for i in range(1, size):
        data += np.add(scalarization_2(file_names[i], regime, q_1, q_2), data)
    for i in range(size):
        data[i] /= size

    np.savetxt(file_name, data)
    return data


def mean(data):
    """
    Calculate mean value of voxels.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, array of mean values of voxels.
    """
    return np.mean(data, axis=1)


def median(data):
    """
    Calculate median of voxels.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, array of median of voxels.
    """
    return np.median(data, axis=1)


def sd(data):
    """
    Calculate standard derivation of voxels.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, array of standard derivation of voxels.
    """
    return np.sd(data, axis=1)


def var(data):
    """
    Calculate variance of voxels.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, array of variance of voxels.
    """
    return np.var(data, axis=1)


def max_min_distance(data):
    """
    Calculate difference between the maximum and minimum of voxel's values.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, difference between the maximum and minimum of voxel's values.
    """
    return np.max(data, axis=1) - np.min(data, axis=1)


def quantiles_distance(data, q_1, q_2):
    """
    Calculate difference between quantiles of voxel's values.

    :param q_1: numeric, level of quantile 1.
    :param q_2: numeric, level of quantile 2.
    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, difference between the quantiles of voxel's values.
    """
    return np.quantile(data, q_2, axis=1) - np.quantile(data, q_1, axis=1)


