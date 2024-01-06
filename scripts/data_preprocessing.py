import numpy as np
import sys
from matplotlib import pyplot as plt 
import pandas as pd

# -------- INSERT PREPROCESSING PROPER --------- # 


# ---------------------------------------------- # 
filepath = '../data/exp.csv'
# this isn't actually "averaged" from the jump, but is the initial dataset TO be averaged, normalized, etc. 
averaged_xy = np.loadtxt(filepath, delimiter=",", dtype='str')
ncol = np.shape(averaged_xy)[1]

# log normalizes the counts
def log_normalize(xy, cols_to_consider:list):
    for col in cols_to_consider: 
        # convert that specific column of the array to a float array, add a pseudocount of one, log normalize it
        xy[:,col]=np.log(np.array(xy[:,col], dtype='float')+1)
    return xy 

# binarizes log-normalized counts to either 0 or 1 based on value 
def clip_df(log_norm_xy, cols_to_consider:list):
    for col in cols_to_consider: 
        c = np.array(log_norm_xy[:,col], dtype='float')
        c[c > 0] = 1
        c[c <= 0] = 0
        log_norm_xy[:,col] = c
    return log_norm_xy

# averages counts over replicates 
def average_replicates(xy, col_range_min:int): 
    averaged_c = np.mean(np.array(xy[:,col_range_min:], dtype='float'), 1)
    xy = xy[:,:col_range_min+1]
    xy[:,col_range_min] = averaged_c
    return xy

# averages replicates, log normalizes counts, clips dataframes
averaged_xy = average_replicates(averaged_xy, ncol-3)
average_ncol = np.shape(averaged_xy)[1]
clipped_average_xy = clip_df(log_normalize(averaged_xy, [average_ncol-1]), [average_ncol-1])
clipped_average_df = pd.DataFrame(clipped_average_xy)

# prints out metrics for my benefit
def record_metrics(clipped_average_xy):
    clipped_dims = np.shape(clipped_average_xy)
    num_ones = len([i for i in clipped_average_xy[:,clipped_dims[1]-1] if float(i) == 1.0]) # <- this can be made more efficient i think
    num_zeros = clipped_dims[0]-num_ones
    print('num ones is: ', num_ones, ' num zeros is: ', num_zeros) # <- log any class imbalance

record_metrics(clipped_average_xy)

def set_splits(type_of_sampling:str, clipped_average_xy, splits:list):
    # set seed to maintain reproducibility
    old_dims = np.shape(clipped_average_xy)
    
    # random splits, using base numpy 
    if type_of_sampling == "random":
        np.random.seed(0)
        np.random.shuffle(clipped_average_xy) # <- shuffle dataset, check if rounding is correct for subsequent steps
    elif type_of_sampling == "balanced": 
        np.random.seed(0)
        last_col = np.array(clipped_average_xy[:,old_dims[1]-1], dtype="float64")
        clipped_average_xy_zeros = np.where(np.equal(last_col,0.0))[0]
        clipped_average_xy_ones = np.where(np.equal(last_col,1.0))[0]
        random_zeros = np.random.choice(clipped_average_xy_zeros, size=500, replace=False)
        random_ones = np.random.choice(clipped_average_xy_ones, size=500, replace=False)
        clipped_average_xy = clipped_average_xy[np.concatenate((random_ones, random_zeros)),:]
    elif type_of_sampling == "stratified": 
        pass 
    
    new_dims = np.shape(clipped_average_xy)

    # TODO: stratified sampling, using sklearn 

    training_range = int(splits[0] * new_dims[0])
    testing_range = training_range + int(splits[1] * new_dims[0])
    validation_range = testing_range + int(splits[2] * new_dims[0])
    # print(training_range, testing_range, validation_range)
    training, testing, validation = clipped_average_xy[:training_range,:], clipped_average_xy[training_range:testing_range,:], clipped_average_xy[testing_range:validation_range,:]

    return training, testing, validation

training, testing, validation = set_splits("balanced", clipped_average_xy, [0.9,0.1,0])
print(np.shape(training), np.shape(testing), np.shape(validation))



# log_norm_xy = log_normalize(xy, [ncol-1,ncol-2,ncol-3])
# log_norm_df = pd.DataFrame(log_norm_xy) # <- just a test dataframe to more easily check/manipulate dimensions 
# print(log_norm_df.head)

# clipped_xy = clip_df(log_norm_xy, [ncol-1,ncol-2,ncol-3])
# clipped_df = pd.DataFrame(clipped_xy) # <- just a test dataframe to more easily check/manipulate dimensions 
# print(clipped_df.head)