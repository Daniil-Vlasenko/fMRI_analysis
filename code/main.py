import dimensionality_reduction_1 as dr
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
import pearson_spearman as cor
import igraph as ig
from sklearn import svm
import preprocessed_data
import synolitic
import processed_data
from pathlib import Path
import nilearn.image as image
import synolitic as syn

# print(len(processed_data.imagery))
# print(len(processed_data.perception))
# print(len(processed_data.imagery_test))
# print(len(processed_data.perception_test))
# print(len(processed_data.imagery_training))
# print(len(processed_data.perception_training))


import classifier_selection
