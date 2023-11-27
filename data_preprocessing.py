import numpy as np
import sys
from matplotlib import pyplot as plt 
import pandas as pd

# -------- INSERT PREPROCESSING PROPER --------- # 


# ---------------------------------------------- # 
filepath = 'exp.csv'
xy = np.loadtxt(filepath, delimiter=",", dtype='str')
averaged_xy = np.loadtxt(filepath, delimiter=",", dtype='str')
print(np.shape(xy)) # (13894, 5)
ncol = np.shape(xy)[1]

def log_normalize(xy, cols_to_consider:list):
    for col in cols_to_consider: 
        xy[:,col]=np.log(np.array(xy[:,col], dtype='float'))
    return xy 

# log_norm_xy = log_normalize(xy, [ncol-1,ncol-2,ncol-3])
# log_norm_df = pd.DataFrame(log_norm_xy) # <- just a test dataframe to more easily check/manipulate dimensions 
# print(log_norm_df.head)

def clip_df(log_norm_xy, cols_to_consider:list):
    for col in cols_to_consider: 
        c = np.array(log_norm_xy[:,col], dtype='float')
        c[c > 0] = 1
        c[c <= 0] = 0
        log_norm_xy[:,col] = c
    return log_norm_xy

# clipped_xy = clip_df(log_norm_xy, [ncol-1,ncol-2,ncol-3])
# clipped_df = pd.DataFrame(clipped_xy) # <- just a test dataframe to more easily check/manipulate dimensions 
# print(clipped_df.head)

def average_replicates(xy, col_range_min:int): 
    averaged_c = np.mean(np.array(xy[:,col_range_min:], dtype='float'), 1)
    xy = xy[:,:col_range_min+1]
    xy[:,col_range_min] = averaged_c
    return xy

averaged_xy = average_replicates(averaged_xy, ncol-3)
average_ncol = np.shape(averaged_xy)[1]
clipped_average_xy = clip_df(log_normalize(averaged_xy, [average_ncol-1]), [average_ncol-1])
clipped_average_df = pd.DataFrame(clipped_average_xy)

def record_metrics(clipped_average_xy):
    clipped_dims = np.shape(clipped_average_xy)
    num_ones = len([i for i in clipped_average_xy[:,clipped_dims[1]-1] if float(i) == 1.0])
    num_zeros = clipped_dims[0]-num_ones
    print('num ones is: ', num_ones, ' num zeros is: ', num_zeros) # <- log any class imbalance

record_metrics(clipped_average_xy)

def set_splits(clipped_average_xy, splits:list):
    # set seed to maintain reproducibility 
    np.random.seed(42)
    xy_dims = np.shape(clipped_average_xy)
    
    # --- random splits, using base numpy 
    np.random.shuffle(clipped_average_xy) # <- shuffle dataset, check if rounding is correct for subsequent steps 
    training_range = int(splits[0] * xy_dims[0])
    testing_range = training_range + int(splits[1] * xy_dims[0])
    validation_range = testing_range + int(splits[2] * xy_dims[0])
    # print(training_range, testing_range, validation_range)
    training, testing, validation = clipped_average_xy[:training_range,:], clipped_average_xy[training_range:testing_range,:], clipped_average_xy[testing_range:validation_range,:]
    
    # --- TODO: stratified sampling, using sklearn 

    return training, testing, validation

training, testing, validation = set_splits(clipped_average_xy, [0.7,0.2,0.1])
print(np.shape(training), np.shape(testing), np.shape(validation))

# ------------ PLOT PROCESSED DISTRIBUTIONS ----------------- # 
# NOTE: i should probably get rid of the header in the file

# setr1_counts = np.array(xy[:,2], dtype='int')
# setr2_counts = np.array(xy[:,3], dtype='int')
# setr3_counts = np.array(xy[:,4], dtype='int')

# # now plot the above counts => distribution of read counts for all conditions 
# f, a = plt.subplots(1, 3, figsize=(10,7.5), layout='constrained')
# f.suptitle('Read Count Distributions')
# print('-------------- setr1 summary stats ------------')
# print('max: ', np.max(setr1_counts), 'min: ', np.min(setr1_counts), 'mean: ', np.mean(setr1_counts), 'median', np.median(setr1_counts))
# a[0].set_xlabel('Frequency')
# a[0].set_ylabel('Read Counts')
# a[0].set_title('Setr1 Read Counts')
# a[0].hist(setr1_counts, bins=np.linspace(1,100))

# print('-------------- setr2 summary stats ------------')
# print('max: ', np.max(setr2_counts), 'min: ', np.min(setr2_counts), 'mean: ', np.mean(setr2_counts), 'median', np.median(setr2_counts))
# a[1].set_xlabel('Frequency')
# a[1].set_ylabel('Read Counts')
# a[1].set_title('Setr2 Read Counts')
# a[1].hist(setr2_counts, bins=np.linspace(1,100))

# print('-------------- setr3 summary stats ------------')
# print('max: ', np.max(setr3_counts), 'min: ', np.min(setr3_counts), 'mean: ', np.mean(setr3_counts), 'median', np.median(setr3_counts))
# a[2].set_xlabel('Frequency')
# a[2].set_ylabel('Read Counts')
# a[2].set_title('Setr3 Read Counts')
# a[2].hist(setr3_counts, bins=np.linspace(1,100))
# # plt.show()
