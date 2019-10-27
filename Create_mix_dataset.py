# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:34:18 2019

@author: Andrea Silva
"""

##############################################################################################################################
#########           MISTURAR O DATASET           #########
import numpy as np

class Mixed_Dataset_Creation():
    def __init__(self, features, output):
        self.features = features
        self.output = output
        
    def insert_features_OUTattribute(self):
        """
        Inserts all the features created and the Out attribute into the final dataset
        Returns the final dataset
        """
        finalDataset=np.array((self.features))
        finalDataset=np.hstack((finalDataset,self.output))
        return finalDataset
    
    def mix_dataset(self, dataset):
        MD=dataset[np.random.permutation(len(dataset))]
        mixedDataset=np.array(MD)
        np.savetxt("../Data/Dataset/Dataset1mixed_WithLimitsOf20and1000.csv",mixedDataset,delimiter=",")
        
    def split_features_OUTattributes(self):
        mixed_dataset=np.genfromtxt("../Data/Dataset/Dataset1mixed_WithLimitsOf20and1000.csv",delimiter=",")
        mixed_features_dataset= np.delete(mixed_dataset, np.s_[len(self.features[0]):], axis=1)  
        mixed_features_dataset.astype(np.float64)
        mixed_OUTattributes_dataset=np.delete(mixed_dataset,np.s_[0:len(self.features[0])],axis=1) 
        mixed_OUTattributes_dataset.astype(np.float64)
        np.savetxt("../Data/Dataset/Features/Features_Dataset_mixed_WithLimitsOf20and1000.csv",mixed_features_dataset, delimiter=",")
        np.savetxt("../Data/Dataset/Out_attributes/Out_Attributes_Dataset_mixed_WithLimitsOf20and1000.csv",mixed_OUTattributes_dataset,delimiter=",")
        print("mixed feature shape")
        print(mixed_features_dataset.shape)
        print("mixed out shape")
        print(mixed_OUTattributes_dataset.shape)
        
##EXECUTAR DATASET_MIXED
print("EXECUTAR DATASET MIXED")
features=np.genfromtxt("../Data/Dataset/Features/Features_Dataset_WithLimitsOf20and1000.csv", delimiter=",")
print("FEATURES LEN: ", len(features))
output=np.genfromtxt("../Data/Dataset/Out_attributes/Is_Transporter_WithLimitsOf20and10000.csv", delimiter=",")
print("OUTPUT LEN: ", len(output))
OUTattributes=output[:,np.newaxis]
mix_data = Mixed_Dataset_Creation(features, OUTattributes)
total_data = mix_data.insert_features_OUTattribute()
mix_data.mix_dataset(total_data)
mix_data.split_features_OUTattributes()
print("MIXED DATASET CRIADO")