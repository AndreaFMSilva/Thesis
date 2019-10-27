# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:01:02 2019

@author: Andrea Silva
"""
##############################################################################################################################
#########           CRIAR OUTPUT           #########
        
import numpy as np
import pandas as pd
class output_dataset():
    def create_IsTransporter_dataset(self, data):
        """
        Creates a numpy array with 1 or 0´s depending if the protein is a transporter(1) or 
        a negative case
        """
        i=True
        index = range(len(data))
        for row in index:
            if i:
                Out_Atribute=np.array((data.values[row][4]))
                i=False
            else:
                Out_Atribute=np.vstack((Out_Atribute,data.values[row][4]))
        return Out_Atribute
    
    


#EXECUTAR A CRIAÇÃO DO OUTPUT
print("CRIANDO O OUTPUT")
data=pd.read_csv("../Data/Dataset/DataDoms_WithLimitsOf20and1000.csv",sep=",")
output_dataset = output_dataset()
IsTransporter = output_dataset.create_IsTransporter_dataset(data)
np.savetxt("../Data/Dataset/Out_attributes/Is_Transporter_WithLimitsOf20and1000.csv",IsTransporter,delimiter=",")
print("OUTPUT CRIADO")
