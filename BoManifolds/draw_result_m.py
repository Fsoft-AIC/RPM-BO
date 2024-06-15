import numpy as np
import pandas as pd
import torch
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ortho_group



def re_order(x):
    a = x
    for i in range(1,len(x)):
        if a[i-1] < a[i]:
            a[i] = a[i-1]
    return a

def transform_dataframe(dataframe):
    df_grouped = (
    dataframe[['index', 'Value']].groupby(['index'])
    .agg(['mean', 'std', 'count'])
)
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
# Calculate a confidence interval as well.
    df_grouped['ci'] = 1.03 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    return df_grouped




 # Compare the result
proj_list = [ "Sphere", "Neural_Net"]
proj_list = ['Sphere']
RunName = ['HeSBO', 'Proposed', 'SSIRBO', 'SIRBO', 'REMBO']#,'SSIRBO2','xREMBO','psiREMBO'
#RunName = ['HeSBO', 'Proposed', 'SSIRBO', 'REMBO']
seed = 0
low_dim = 10
high_dim = 1000
d_embedding_list = [8, 15, 20]
box_size = 1
np.random.seed(seed)
matrix = ortho_group.rvs(dim=high_dim)
matrix_basis = matrix[:,:low_dim]
centroid = np.zeros(low_dim)
noise_subspace = np.zeros(high_dim)
hyper_opt_interval = 3
radius = 1
start_rep = 1
stop_rep = 5
test_func = "Ackley_Sphere_1" #MNIST Branin Hartmann6
total_iter = 300
initial_n = 10
variance = 0
raw_samples =  100#512
num_restarts = 7#10
m = 7
batch_size = 1 # The number of extreme points that acquisition chooses
acquisition_function_list = ["EI"]
embedding_boundaries_setting = ["constant"]

    # Create data to plot
rembo_value = []
proposed_model_value = []
dataframe_list = []


    # Get the initial points of HesBO
    # file fix initial points
filename_fix = '_' + str(test_func) +'_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + '_seed' + str(seed)+'.csv'
    
m = 9
    # Read the result from csv file

dataframe_list = []


for model in RunName:
    if model == 'Proposed':
        for type in proj_list:
            for acq in acquisition_function_list:
                for embedding_boundaries in embedding_boundaries_setting:
                    for d_embedding in d_embedding_list:
                        file_name = str(test_func)+'_acq' + str(acq) + '_style' + str(embedding_boundaries) + '_proj' + str(type)+'_result_proposed_d'+ str(low_dim) + '_D' + str(high_dim) + '_m' + str(
                                                d_embedding) + '_iter' + str(total_iter + initial_n) + '_batchsize' + str(batch_size)+ '_restart' + str(stop_rep - start_rep + 1) +'_seed' + str(seed) +'.csv'
                        df = pd.read_csv(file_name)
                        len_result = len(df)
                        dataframe_proposed = pd.DataFrame()
                        for i in range(len_result):
                            df1 = pd.DataFrame(range(1,(total_iter+initial_n) +1 - m), columns = ["index"])
                            df1["Type"] = "Proposed_" + str(acq) + '_' + str(embedding_boundaries) + '_type_' + str(type) + '_embedded_dim_' + str(d_embedding)
                            df1["Value"] = re_order(df.iloc[i].to_numpy())[m:] # proposed_model_value[i] 
                            dataframe_proposed = pd.concat([dataframe_proposed, df1], axis = 0, ignore_index = True)
                        dataframe_list.append(dataframe_proposed)
    else:
        filename = str(model) + filename_fix
        df = pd.read_csv(filename)
        len_result = len(df)
        dataframe = pd.DataFrame()
        for i in range(len_result):
            df1 = pd.DataFrame(range(1,(total_iter+initial_n) +1 - m), columns = ["index"])
            df1["Type"] = str(model) 
            df1["Value"] = re_order(-df.iloc[i].to_numpy())[m:] # proposed_model_value[i] 
            dataframe = pd.concat([dataframe, df1], axis = 0, ignore_index = True)
        dataframe_list.append(dataframe)



colors = ['blue', 'orange', 'green', 'red', 'black', 'cyan', 'magenta']
plt.figure(figsize=(14,7))
for i in range(len(dataframe_list)):
    df_grouped = transform_dataframe(dataframe_list[i])
    x = df_grouped['index']
    plt.plot(x, df_grouped['mean'], color = colors[i], label = dataframe_list[i]['Type'].unique()[0])
    plt.fill_between(
                x, df_grouped['ci_lower'], df_grouped['ci_upper'], color=colors[i], alpha=.15)
plt.legend()
plt.title(str(test_func) +" function with latent effective sphere subspace " + str(high_dim) + "-" + str(low_dim))
plt.show()
