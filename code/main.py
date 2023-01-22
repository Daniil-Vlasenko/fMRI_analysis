import dimensionality_reduction as dr
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
import pearson_spearman as cor
import igraph as ig




img = nib.load("../processed_data/sub-01_ses-perceptionTraining01_task-perception_run-01_bold_preproc_10_10_10.nii")
data = img.get_fdata()
print(data.shape)
new_data = dr._4D_to_2D(data)
print(new_data.shape)

print(new_data)
print(np.mean(new_data, axis=1))



