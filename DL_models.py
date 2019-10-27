# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 01:17:05 2019

@author: Andrea Silva
"""


##############################################################################################################################
#########           DL MODELS           #########
import tensorflow as tf
#tf.enable_eager_execution()

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

TF_CONFIG_ = tf.ConfigProto()

TF_CONFIG_.gpu_options.allow_growth = True

sess = tf.Session(config = TF_CONFIG_)

tf.keras.backend.set_session(sess)

import pandas as pd

class dl_models():   
    def rnn_model(self):    
        Input = pd.read_csv("Data_mixed_seqs_WithLimitsOf20and1000.csv", delimiter=",", header = None)   #X
        print("Input ",Input)
        print("LEN :", len(Input))
        
        Output = pd.read_csv("Out_Attributes_Dataset_mixed_WithLimitsOf20and1000.csv", delimiter=",", header = None)
        print("Output", Output)
        print("LEN :", len(Output))
        
        max_len = 0
        min_len = 10000
        samples = []
        for i in Input.iloc[:,0]:
            samples.append(i)
            if len(i) > max_len:
                max_len = len(i)
            elif len(i) < min_len:
                min_len = len(i)
        print(max_len)
        print(min_len)
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level = True)
        tokenizer.fit_on_texts(samples)
        sequences = tokenizer.texts_to_sequences(samples)
        
        data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = max_len)    #data = pad_sequences(sequences, maxlen=maxlen)
        
        one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        
        
        
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
        
        
        print("\n    MODELO 1    \n")
        model = tf.keras.Sequential([
                tf.keras.layers.Embedding(len(word_index)+1, 128),
                tf.keras.layers.CuDNNLSTM(32),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])    
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 2    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.CuDNNGRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 3    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 4    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ]) 
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy']) 
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 5    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(8)),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy']) 
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 6    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 7    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 5),
            tf.keras.layers.MaxPooling1D(3),
            tf.keras.layers.Conv1D(32, 5),
            tf.keras.layers.MaxPooling1D(3),
            tf.keras.layers.Conv1D(32, 5),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 8    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.CuDNNGRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 9    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 10    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.CuDNNLSTM(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])                         
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 11    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 12    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.CuDNNLSTM(32),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 13    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.CuDNNLSTM(16, return_sequences=True),
            tf.keras.layers.CuDNNLSTM(16, return_sequences=True),
            tf.keras.layers.CuDNNLSTM(16),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 14    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.CuDNNGRU(32, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.CuDNNGRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 15    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.CuDNNGRU(128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 16    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.CuDNNLSTM(128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 17    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.CuDNNLSTM(128, return_sequences = True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.CuDNNLSTM(128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 18    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.GRU(128),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 19    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(64, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(64, 7),
            tf.keras.layers.GRU(32, return_sequences = True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 20    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=20, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 21    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.CuDNNGRU(128, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.CuDNNGRU(128),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=20, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 22    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.GRU(128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=20, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 21  -- COM + DROPOUT    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(128, 7),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.CuDNNGRU(128, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.CuDNNGRU(128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=20, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        
        
        print("\n    MODELO 20  -- COM DROPOUT   \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv1D(32, 7),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=20, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
        print("\n    MODELO 23    \n")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(word_index)+1, 128),
            tf.keras.layers.Conv1D(256, 7),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(256, 7),
            tf.keras.layers.CuDNNGRU(256, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.CuDNNGRU(256),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(input_train, output_train, epochs=10, batch_size=128, validation_data = (input_val, output_val))
        res = model.evaluate(input_test, output_test)
        print(res)
        
##EXECUTAR ALL DL MODELS
rnn = dl_models()
rnn.rnn_model()