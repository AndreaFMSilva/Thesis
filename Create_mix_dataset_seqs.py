# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:39:23 2019

@author: Andrea Silva
"""

##############################################################################################################################
#########           CRIAR O DATASET COM AS SEQUENCIAS DAS PROTEINAS, MISTURADO           #########
from numpy import genfromtxt
import numpy as np
import pandas
class Dataset_Mixed_Seqs():

    def load_files(self): 
        #carrega os ficheiros 
        Input = genfromtxt("../Data/Dataset/Features/Features_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")   #X
        Output = genfromtxt("../Data/Dataset/Out_attributes/Out_Attributes_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")   #Y
        DataDoms = pandas.read_csv("../Data/Dataset/DataDoms_WithLimitsOf20and1000.csv", delimiter=",")
        return Input, Output, DataDoms
    
    def obtain_seqs(self):       
        Input, Output, DataDoms = self.load_files()
        
        ids = []
        for i in Input[:,0]:
            ids.append(int(i))
        print(len(ids))


        seqs = np.array(("Sequencias"))
        for i in ids:
            seqs = np.vstack((seqs, DataDoms.iloc[i,3]))

        df=pandas.DataFrame(seqs[1:,], index=None)
        df.to_csv("../Data/Dataset/Data_mixed_seqs_WithLimitsOf20and1000.csv", sep=',', index=None, header = None)

#EXECUTAR DATASET MIXED SEQS
mix_data_seqs = Dataset_Mixed_Seqs()
mix_data_seqs.obtain_seqs()