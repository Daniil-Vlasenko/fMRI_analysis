import nibabel as nib
import numpy as np
import pandas as pd


def set_generation(input_file, output_file):
    """
    Generate or extend the file that contain values of voxels.

    :param input_file: string, name of input .nii file.
    :param output_file: string, name of output .txt file.
    :return:
    """


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
    return np.quantile(data, q_2) - np.quantile(data, q_1)
