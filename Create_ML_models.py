# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:43:39 2019

@author: Andrea Silva
"""


##############################################################################################################################
#########           CREATE BEST MACHINE LEARNING MODEL           #########
#from sklearn import model_selection
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
#from sklearn import preprocessing
#from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from numpy import genfromtxt
import os
import joblib


class ML_models():
    def load_and_treat_files(self): 
        #carrega os ficheiros 
        Input = genfromtxt("../Data/Dataset/Features/Features_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")   #X
        Output = genfromtxt("../Data/Dataset/Out_attributes/Out_Attributes_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")   #Y

        directory="../Data/ML Models/"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        Input = Input[:,1:]
        print("INPUT")
        print(Input)
        print("OUTPUT")
        print(Output)
        
        print(Input.shape)
        #Variabilidade
        sel    =    VarianceThreshold()  
        print("SEL \t", sel)
        filt    =    sel.fit_transform (Input)    
        print("FILT  \t", filt)
        print(filt.shape)
        print("VarianceThreshold filter used")
        
        print("Aplicação do Imputer")
        imputer = Imputer()
        data = imputer.fit_transform(filt)
        print(data)
        print(data.shape)
        
        return data, Output
    
    def create_models(self, data, Output):
        directory="../Data/ML Models/"
        
        input_train = data[:13836]
        print("INPUT TRAIN SHAPE: ", input_train.shape)
        output_train = Output[:13836]
        print("OUTPUT TRAIN SHAPE: ", output_train.shape)
#        input_val = data[13836:23836]
#        print("INPUT VAL SHAPE: ", input_val.shape)
#        output_val = Output[13836:23836]
#        print("OUTPUT VAL SHAPE: ", output_val.shape)
        input_test = data[13836:]
        print("INPUT TEST SHAPE: ", input_test.shape)
        output_test = Output[13836:]
        print("OUTPUT TEST SHAPE: ", output_test.shape)

        NB_model= BaggingClassifier(GaussianNB())
        print("NB_model \t", NB_model)
        tree_model = BaggingClassifier(ExtraTreesClassifier())
        print("Tree_model \t", tree_model)
        knn = BaggingClassifier(KNeighborsClassifier())
        print("KNN \t", knn)
        logistic = BaggingClassifier(linear_model.LogisticRegression())
        print("Logistic \t", logistic)
        GB_model= GradientBoostingClassifier()
        print("GB_model \t", GB_model)
        RF_model= BaggingClassifier(RandomForestClassifier())
        print("RF_model \t", RF_model)
        svm_model = svm.SVC()
        print("SVM_model \t", svm_model)
        
        print("        NAIVE BAYES MODEL        \n")
        NB_model.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,NB_model.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,NB_model.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,NB_model.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,NB_model.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,NB_model.predict(input_test)))
        
        #Saving the NB model 
        learned_NB=directory+"/NB_model.pkl"
        joblib.dump(NB_model,learned_NB)
        
        
        print("\n        EXTRATREECLASSIFIER MODEL        \n")
        tree_model.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,tree_model.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,tree_model.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,tree_model.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,tree_model.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,tree_model.predict(input_test)))
        
        #Saving the Tree model
        learned_tree=directory+"/Tree_model.pkl"
        joblib.dump(tree_model,learned_tree)
        
        
        print("\n        K NEAREST NEIGHBOURS MODEL        \n")
        knn.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,knn.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,knn.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,knn.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,knn.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,knn.predict(input_test)))
        
        #Saving the knn model 
        learned_knn=directory+"/Knn_model.pkl"
        joblib.dump(knn,learned_knn)
        
        
        print("\n        LOGISTIC REGRESSION MODEL        \n")
        logistic.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,logistic.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,logistic.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,logistic.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,logistic.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,logistic.predict(input_test)))
        
        #Saving the logistic regression model
        learned_logistic=directory+"/Logistic_model.pkl"
        joblib.dump(logistic,learned_logistic)
                

        print("\n        GRADIENT BOOSTING MODEL        \n")
        GB_model.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,GB_model.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,GB_model.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,GB_model.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,GB_model.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,GB_model.predict(input_test)))
        
        #Saving the GB model 
        learned_GB=directory+"/GB_model.pkl"
        joblib.dump(GB_model,learned_GB)

        
        print("\n        RANDOM FOREST MODEL        \n")
        RF_model.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,RF_model.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,RF_model.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,RF_model.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,RF_model.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,RF_model.predict(input_test)))
        
        #Saving the RF model 
        learned_RF=directory+"/RF_model.pkl"
        joblib.dump(RF_model,learned_RF)

        
        print("\n        SVM MODEL        \n")
        svm_model.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test,svm_model.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,svm_model.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,svm_model.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,svm_model.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,svm_model.predict(input_test)))
        
        #Saving the SVM model
        learned_svm=directory+"/SVM_model.pkl"
        joblib.dump(svm_model,learned_svm)

    def create_voting_classifiers(self, data, Output):
        NB_model= BaggingClassifier(GaussianNB())
        print("NB_model \t", NB_model)
        tree_model = BaggingClassifier(ExtraTreesClassifier())
        print("Tree_model \t", tree_model)
        knn = BaggingClassifier(KNeighborsClassifier())
        print("KNN \t", knn)
        logistic = BaggingClassifier(linear_model.LogisticRegression())
        print("Logistic \t", logistic)
        GB_model= GradientBoostingClassifier()
        print("GB_model \t", GB_model)
        RF_model= BaggingClassifier(RandomForestClassifier())
        print("RF_model \t", RF_model)
        svm_model = svm.SVC()
        print("SVM_model \t", svm_model)
        
        input_train = data[:13836]
        print("INPUT TRAIN SHAPE: ", input_train.shape)
        output_train = Output[:13836]
        print("OUTPUT TRAIN SHAPE: ", output_train.shape)
        input_val = data[13836:23836]
        print("INPUT VAL SHAPE: ", input_val.shape)
        output_val = Output[13836:23836]
        print("OUTPUT VAL SHAPE: ", output_val.shape)
        input_test = data[23836:]
        print("INPUT TEST SHAPE: ", input_test.shape)
        output_test = Output[23836:]
        print("OUTPUT TEST SHAPE: ", output_test.shape)
        
        print("\n HARD VOTE WITHOUT WEIGHTS  \n")
        HardVote=VotingClassifier(estimators=[('NB',NB_model),('tree',tree_model),('knn',knn),('logistic',logistic),('GB',GB_model),('RF',RF_model),('SVM',svm_model)],voting='hard')
        HardVote.fit(input_train, output_train)
        accuracy = HardVote.score(input_test,output_test)
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,HardVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,HardVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,HardVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,HardVote.predict(input_test)))
        
        print("\n  HARD VOTE WITH WEIGHTS: [1,2,3,2,3,3,1] \n")
        HardVote=VotingClassifier(estimators=[('NB',NB_model),('tree',tree_model),('knn',knn),('logistic',logistic),('GB',GB_model),('RF',RF_model),('SVM',svm_model)],voting='hard',weights=[1,2,3,2,3,3,1])
        HardVote.fit(input_train, output_train)
        accuracy = HardVote.score(input_test,output_test)
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,HardVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,HardVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,HardVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,HardVote.predict(input_test)))
        
        print("\n HARD VOTE WITH WEIGHTS: [4,6,10,6,10,7,1] \n")
        HardVote=VotingClassifier(estimators=[('NB',NB_model),('tree',tree_model),('knn',knn),('logistic',logistic),('GB',GB_model),('RF',RF_model),('SVM',svm_model)],voting='hard',weights=[4,6,10,6,10,7,1])
        HardVote.fit(input_train, output_train)
        accuracy = HardVote.score(input_test,output_test)
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,HardVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,HardVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,HardVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,HardVote.predict(input_test)))
        
        print("\n HARD VOTE WITHOUT WEIGHTS BUT ONLY WITH GB, KNN AND RF \n" )
        HardVote=VotingClassifier(estimators=[('knn',knn),('GB',GB_model),('RF',RF_model)],voting='hard')
        HardVote.fit(input_train, output_train)
        accuracy = HardVote.score(input_test,output_test)
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,HardVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,HardVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,HardVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,HardVote.predict(input_test)))
        
        print("\n BEST WEIGHTED HARD VOTE : [3,3,10,3,10,10,1] \n")
        BestHardVote=VotingClassifier(estimators=[('NB',NB_model),('tree',tree_model),('knn',knn),('logistic',logistic),('GB',GB_model),('RF',RF_model),('SVM',svm_model)],voting='hard',weights=[3,3,10,3,10,10,1])
        BestHardVote.fit(input_train, output_train)
        accuracy = BestHardVote.score(input_test,output_test)
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,BestHardVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,BestHardVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,BestHardVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,BestHardVote.predict(input_test)))
        
        svm_model_prob_true = svm.SVC(probability=True)
        
        print("\n SOFT VOTE : [1,2,3,2,3,3,1] \n")
        SoftVote=VotingClassifier(estimators=[('NB',NB_model),('tree',tree_model),('knn',knn),('logistic',logistic),('GB',GB_model),('RF',RF_model),('SVM',svm_model_prob_true)],voting='soft',weights=[1,2,3,2,3,3,1])
        SoftVote.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test, SoftVote.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,SoftVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,SoftVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,SoftVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,SoftVote.predict(input_test)))
        
        print("\n SOFT VOTE : [1,2,3,2,3,3,1] \n")
        SoftVote=VotingClassifier(estimators=[('NB',NB_model),('tree',tree_model),('knn',knn),('logistic',logistic),('GB',GB_model),('RF',RF_model),('SVM',svm_model_prob_true)],voting='soft')
        SoftVote.fit(input_train, output_train)
        accuracy = metrics.accuracy_score(output_test, SoftVote.predict(input_test))
        print("ACCURACY: ", accuracy)
        roc_auc = metrics.roc_auc_score(output_test,SoftVote.predict(input_test))
        print("ROC_AUC: ", roc_auc)
        f1= metrics.f1_score(output_test,SoftVote.predict(input_test))
        print("F1: ", f1)
        recall = metrics.recall_score(output_test,SoftVote.predict(input_test))
        print("RECALL: ", recall)
        print("MATRIZ CONFUSÃO: ", metrics.confusion_matrix(output_test,SoftVote.predict(input_test)))

##EXECUTAR ML MODELS
ml_models = ML_models()
data,Output = ml_models.load_and_treat_files()
ml_models.create_models(data,Output)
ml_models.create_voting_classifiers(data,Output)