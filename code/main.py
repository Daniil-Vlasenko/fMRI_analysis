import dimensionality_reduction as dr
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
import pearson_spearman as cor
import igraph as ig
import synolitic
import data

print(len(data.imagery))
print(len(data.perception))
print(len(data.imagery_test))
print(len(data.perception_test))
print(len(data.imagery_training))
print(len(data.perception_training))

for i in data.all_files:
    open(i)









