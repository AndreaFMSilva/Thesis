# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:52:12 2019

@author: Andrea Silva
"""

##############################################################################################################################
#########           HYPERPARAMETERS OPTIMIZATION           #########
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.ensemble import GradientBoostingClassifier
from numpy import genfromtxt
from sklearn import metrics

class hyperparameters_optimization():
    def optimize(self, parameters, n_iter=50):
        Input = genfromtxt("../Data/Dataset/Features/Features_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")[:,1:]   #X
        print("INPUT \n", Input)
        print(Input.shape)
        Output = genfromtxt("../Data/Dataset/Out_attributes/Out_Attributes_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")   #Y
        print("OUTPUT \n", Output)
        print(Output.shape)
        
        input_train = Input[:13836]
        print("INPUT TRAIN SHAPE: ", input_train.shape)
        output_train = Output[:13836]
        print("OUTPUT TRAIN SHAPE: ", output_train.shape)
        input_val = Input[13836:23836]
        print("INPUT VAL SHAPE: ", input_val.shape)
        output_val = Output[13836:23836]
        print("OUTPUT VAL SHAPE: ", output_val.shape)
        input_test = Input[23836:]
        print("INPUT TEST SHAPE: ", input_test.shape)
        output_test = Output[23836:]
        print("OUTPUT TEST SHAPE: ", output_test.shape)
                
        GB_model= GradientBoostingClassifier()
        GB_model.fit(input_train, output_train)
        
        #Procura Aleatória, com validação cruzada na estimação do erro
        rand_search = RandomizedSearchCV(GB_model, param_distributions=parameters, n_iter)
        rand_search.fit(input_val, output_val) 
        
        accuracy = metrics.accuracy_score(output_test, rand_search.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,rand_search.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,rand_search.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,rand_search.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,rand_search.predict(input_test)))


#EXECUTAR HYPERPARAMETERS OPTIMIZATION
parameters = {'n_estimators':[50,100,500],'learning_rate':[0.05, 0.1, 0.2], 'min_samples_split':[2,10, 100, 300, 500],
              'min_samples_leaf':[1,10,50], 'max_depth':[3,5,8]}
hyp_opt = hyperparameters_optimization()
hyp_opt.optimize(parameters)