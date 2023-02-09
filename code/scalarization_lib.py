import nibabel as nib
import numpy as np
import pandas as pd
import dimensionality_reduction_1_lib as dr


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
        return std(data)
    elif regime == 3:
        return var(data)
    elif regime == 4:
        return max_min_distance(data)
    elif regime == 5:
        return quantiles_distance(data, q_1, q_2)
    elif regime == 6:
        return max_(data)
    elif regime == 7:
        return min_(data)
    elif regime == 8:
        return quantile_1(data, q_1)
    elif regime == 9:
        return quantile_2(data, q_2)
    else:
        raise Exception("Invalid regime.")


def scalarization_2(nifti_file, regime, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value.

    :param nifti_file: string, name of input file.
    :param regime: numeric, regime of function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance,
    4 - difference between the maximum and minimum, 5 - difference between quantiles.
    :param q_1: numeric, level of quantile 1.
    :param q_2:numeric, level of quantile 2.
    :return: ndarray, array of scalars.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    data = dr._4D_to_2D(data)
    return scalarization_1(data, regime, q_1, q_2)


def scalarization_3(nifti_files, scalars_file, regime, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value and save them to file.

    :param nifti_files: list of strings, names of input files.
    :param scalars_file: string, name of output file.
    :param regime: numeric, regime of function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance,
    4 - difference between the maximum and minimum, 5 - difference between quantiles.
    :param q_1: numeric, level of quantile 1.
    :param q_2:numeric, level of quantile 2.
    :return: ndarray, array of scalars.
    """
    data = scalarization_2(nifti_files[0], regime)
    size = len(nifti_files)
    for i in range(1, size):
        data = np.vstack((data, scalarization_2(nifti_files[i], regime, q_1, q_2)))
    data = np.transpose(data)
    np.savetxt(scalars_file, data)
    return data


# def scalarization_4(nifti_per_files, nifti_im_files, scalars_per_file, scalars_im_file, q_1=0.1, q_2=0.9):
#     for i in range(10):
#         scalarization_3(nifti_per_files, scalars_per_file[]
#                         "../correlations/test/dimensionality_reduction_1/10_10_10/synolitic_method_1/scalars/imagery/mean.txt",
#                         0)


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


def std(data):
    """
    Calculate standard derivation of voxels.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, array of standard derivation of voxels.
    """
    return np.std(data, axis=1)


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


def max_(data):
    """
    Calculate maximum of voxel's values.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, maximum voxel's values.
    """
    return np.max(data, axis=1)


def min_(data):
    """
    Calculate minimum of voxel's values.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :return: ndarray, minimum voxel's values.
    """
    return np.min(data, axis=1)


def quantile_1(data, q_1):
    """
    Calculate quantile of voxel's values.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :param q_1: numeric, level of quantile.
    :return: ndarray, quantile voxel's values.
    """
    return np.quantile(data, q_1, axis=1)

def quantile_2(data, q_2):
    """
    Calculate quantile of voxel's values.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :param q_2: numeric, level of quantile.
    :return: ndarray, quantile voxel's values.
    """
    return np.quantile(data, q_2, axis=1)


