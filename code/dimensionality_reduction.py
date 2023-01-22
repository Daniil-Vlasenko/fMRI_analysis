import nilearn.image as image
import numpy as np


def dimensionality_reduction_1(img_input, n):
    """
    Dimensionality reduction by downsample img_input's grid to n millimeters per voxel.

    :param img_input: niimg-like object.
    :param n: int, number of millimeters per voxel.
    :return: nibabel.Nifti1Image.
    """
    img_input_affine = img_input.affine.copy()
    for i in range(3):
        img_input_affine[i][i] = n * np.sign(img_input_affine[i][i])
    return image.resample_img(img_input, img_input_affine)


def _4D_to_2D(data):
    """
    Сonverting 4D array to array of arrays that contains voxel's values at different time.

    :param data: ndarray, 4D array.
    :return: ndarray, array of arrays that contain voxel's values at different time.
    """

    return data.transpose(3, 0, 1, 2).reshape(data.shape[3], -1).transpose(1, 0)


def _2D_to_4D(data, shape):
    """
    Сonverting array of arrays that contains voxel's values at different time to 4D array.

    :param data: ndarray, array of arrays that contain voxel's values at different time.
    :param shape: list, shape of 4D array.
    :return: ndarray, 4D array.
    """

    return data.reshape(shape)


def delete_empty_voxels_1(data, eps):
    """
    Add to every element in array of arrays that contains voxel's values at different time id of the voxel and
    then delete voxels which value at time t=0 less than eps.

    :param data: ndarray, array of arrays that contains voxel's values at different time.
    :param eps: numeric.
    :return: array of tuples (id of voxel, values of voxel in different time).
    """
    result = []
    for id in range(data.shape[0]):
        if data[id][0] >= eps:
            result.append((id, data[id]))
    return result


def delete_empty_voxels_2(data, eps):
    """
    Add to every element in array of arrays that contains voxel's values at different time id of the voxel and
    then delete voxels which value at time t=0 less than eps.

    :param data: ndarray, array of arrays that contains voxel's values at different time.
    :param eps: numeric.
    :return: array of voxels which value at time t=0 greater than eps.
    """
    result = []
    for id in range(data.shape[0]):
        if data[id][0] >= eps:
            result.append(data[id])
    return result


def delete_empty_voxels_3(data, eps):
    """
    Delete voxels which value at time t=0 less than eps.

    :param data: ndarray, array of arrays that contains voxel's values at different time.
    :param eps: numeric.
    :return: (array of voxels which value at time t=0 greater than eps, voxel's values at different time)
    """
    result_1, result_2 = [], []
    for id in range(data.shape[0]):
        if data[id][0] >= eps:
            result_1.append(id)
            result_2.append(data[id])
    return result_1, result_2
