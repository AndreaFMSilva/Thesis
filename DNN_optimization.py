# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:48:31 2019

@author: Andrea Silva
"""

##############################################################################################################################
#########           CREATE DNNS AND OPTIMIZATION           #########
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from random import choice
from numpy import genfromtxt


class DNN_optimization():   
    def __init__(self):
        Input = genfromtxt("../Data/Dataset/Features/Features_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")[:,1:]   #X
        print("INPUT \n", Input)
        print(Input.shape)
        Output = genfromtxt("../Data/Dataset/Out_attributes/Out_Attributes_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",")   #Y
        print("OUTPUT \n", Output)
        print(Output.shape)
        
        self.input_train = Input[:13836]
        print("INPUT TRAIN SHAPE: ", self.input_train.shape)
        self.output_train = Output[:13836]
        print("OUTPUT TRAIN SHAPE: ", self.output_train.shape)
        self.input_val = Input[13836:23836]
        print("INPUT VAL SHAPE: ", self.input_val.shape)
        self.output_val = Output[13836:23836]
        print("OUTPUT VAL SHAPE: ", self.output_val.shape)
        self.input_test = Input[23836:]
        print("INPUT TEST SHAPE: ", self.input_test.shape)
        self.output_test = Output[23836:]
        print("OUTPUT TEST SHAPE: ", self.output_test.shape)
    
    def setup_model(self, topo, dropout_rate, input_size, output_size):

        model = Sequential()    
        model.add(Dense(topo[0], activation="relu", input_dim = input_size))
        if dropout_rate > 0: model.add(Dropout(dropout_rate))
        for i in range(1,len(topo)):        
            model.add(Dense(topo[i], activation="relu"))
            if dropout_rate > 0: model.add(Dropout(dropout_rate))    
        model.add(Dense(output_size, activation = 'sigmoid'))

        return model
    
    def train_dnn(self, model, alg, epochs = 50, batch_size = 512):
        if alg == "adam":
            optimizer = optimizers.Adam()
        elif alg == "rmsprop":
            optimizer = optimizers.RMSprop()
        elif alg == "sgd_momentum":
            optimizer = optimizers.SGD()
        else: 
            optimizer = optimizers.SGD()

        model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
        model.fit(self.input_train, self.output_train, epochs = epochs, batch_size = batch_size, verbose = 0) 
        
        return model
    
    def dnn_optimization(self, opt_params, iterations = 85, verbose = True):
        if verbose: 
            print("Topology\tDropout\tAlgorithm\tLRate\tValLoss\tValAcc\n")
        best_acc = None

        input_size = self.input_train.shape[1]
        output_size = 1

        if "topology" in opt_params:
            topologies = opt_params["topology"]
        else: topologies = [[100]]
        if "algorithm" in opt_params:
            algs = opt_params["algorithm"]
        else: algs = ["adam"]
        if "dropout" in opt_params:
            dropouts = opt_params["dropout"]
        else: dropouts= [0.0]

        for it in range(iterations):
            topo = choice(topologies)
            dropout_rate = choice(dropouts)
            dnn = self.setup_model (topo, dropout_rate, input_size, output_size)
            alg = choice(algs)
            dnn = self.train_dnn(dnn, alg)
            val_loss, val_acc = dnn.evaluate(self.input_test, self.output_test, verbose = 0)

            if verbose: 
                print(topo, "\t", dropout_rate, "\t", alg, "\t", "\t", val_loss, "\t", val_acc)

            if best_acc is None or val_acc > best_acc:
                best_acc = val_acc
                best_config = (topo, dropout_rate, alg)

        return best_config, best_acc

##EXECUTAR DNN
dnn = DNN_optimization()
opt_pars = {"topology":[[128], [128,128], [128,64], [128,128,128,128], [256], [256,256], [256,128], [256,256,256,256], [256,128,128,256]],
            "algorithm": [ "adam", "rmsprop", "sgd_momentum"],
            "dropout": [0, 0.2, 0.5]}

best_config, best_acc = dnn.dnn_optimization(opt_pars)  
print("BEST CONFIG: ",best_config)
print("BEST ACC: ",best_acc)
        
    