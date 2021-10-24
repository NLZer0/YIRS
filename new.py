import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data/iris.csv')

species_list = list(df.Species.unique())
species_dict = dict(zip(species_list, range(6)))
species_id = np.zeros(df.Id.count())
for i in range(species_id.shape[0]):
    species_id[i] = species_dict[df.Species.iloc[i]]
df['Species'] = species_id.astype(int)

def calc_metric(df):
    inter_sс_matrix = df.corr().values
    
    unique_classes = np.array([0,1,2])
    in_sс_matrix = np.zeros((3, inter_sс_matrix.shape[0]-1, inter_sс_matrix.shape[1]-1))
    for i in unique_classes:
        df_class_i = df.loc[df.Species == i]
        in_sс_matrix[i] = df_class_i.corr().values[:-1, :-1]
    
    in_sс_matrix_mean = np.mean(in_sс_matrix, axis=0)
    
    R_tr = np.sum(np.diag(inter_sс_matrix,1))
    R1_tr = np.sum(np.diag(in_sс_matrix_mean,1))
    criterion = R_tr/R1_tr
    return criterion

print(calc_metric(df.loc[:, ['PetalLengthCm', 'Species']]))