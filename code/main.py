import dimensionality_reduction_1 as dr
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
import pearson_spearman as cor
import igraph as ig
import synolitic
import preprocessed_data
from pathlib import Path
import nilearn.image as image


print(len(data.imagery))
print(len(data.perception))
print(len(data.imagery_test))
print(len(data.perception_test))
print(len(data.imagery_training))
print(len(data.perception_training))

iput_files = data.all_files
n = 10
for i in iput_files:
    img = image.load_img(i)
    affine = img.affine.copy()
    print(i)
    print(affine)
    for j in range(3):
        affine[j][j] = n * np.sign(affine[j][j])
    new_img = image.resample_img(img, affine)
    path_array = i.split("/")
    new_path = "../processed_data/dimensionality_reduction_1/" + path_array[5] + "/" + path_array[6] + "/" + path_array[
        8]
    nib.save(new_img, new_path)












