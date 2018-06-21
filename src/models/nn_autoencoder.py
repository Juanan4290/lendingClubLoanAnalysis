#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:43:10 2018

@author: Juan Antonio Morales
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessing_functions import normalize_variables
from src.model_launcher import model_evaluation

def nn_autoencoder(data, n_hidden_1, n_hidden_2, epochs, batch_size, learning_rate,
                   normalization = "standard"):
    """
    Parameters
    ---------
    data: DataFrame to encode and fit a logistic regression
    n_hidden_1: number of nuerons in first hidden layer
    n_hidden_2: number of nuerons in second hidden layer (encoded)
    epochs: number of forward and backward passes of all the training examples
    batch_size: number of training examples in one forward/backward pass
    learning_rate: learning rate hyperparameter for updating weights
    normalization: type of normalization to perform: "robust", "standard" and "minMax"
    
    Returns
    ---------
    result: logistic regression model and evaluation model with encoded data: 
            AUC in train and test, confusion matrix, accuracy, 
            recall and precision (treshold = 0.5)
    """
    
    ### pre-process
    print("Preprocessing...")
    # normalization
    data = normalize_variables(data, normalization)
        
    # target
    X = data.loc[:, data.columns!='loan_status']
    y = data['loan_status']
    
    ### autoencoder for feature extraction
    print("Training autoencoder for feature extraction...")
    
    # batch size split
    X_split = X.values

    num_batches = int(X_split.shape[0] / batch_size)
    X_split = np.array_split(X_split, num_batches)
    
    ### neural network architecture
    # hidden layers
    _, number_of_variables = np.shape(X_split[0])
    num_hidden_0 = number_of_variables
    num_hidden_1 = n_hidden_1
    num_hidden_2 = n_hidden_2
    
    # parameters initilization
    x = tf.placeholder(dtype = tf.float64, shape = [None, num_hidden_0])
    
    weights = {
        # encoder
        'w1': tf.Variable(tf.truncated_normal(stddev=.1, shape=[num_hidden_0, num_hidden_1], 
                                              dtype=tf.float64)),
        'w2': tf.Variable(tf.truncated_normal(stddev=.1, shape=[num_hidden_1, num_hidden_2], 
                                              dtype=tf.float64)),
        # decoder
        'w3': tf.Variable(tf.truncated_normal(stddev=.1, shape=[num_hidden_2, num_hidden_1], 
                                              dtype=tf.float64)),
        'w4': tf.Variable(tf.truncated_normal(stddev=.1, shape=[num_hidden_1, num_hidden_0], 
                                              dtype=tf.float64)),
    }
    
    biases = {
        # encoder
        'b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
        'b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
        # decoder
        'b3': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
        'b4': tf.Variable(tf.random_normal([num_hidden_0], dtype=tf.float64)),
    }
    
    # foward propagation
    hidden_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    hidden_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_1, weights['w2']), biases['b2']))
    hidden_3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_2, weights['w3']), biases['b3']))
    autoencoded = tf.nn.sigmoid(tf.add(tf.matmul(hidden_3, weights['w4']), biases['b4']))
    
    # loss function
    loss = tf.reduce_mean(tf.pow(x - autoencoded, 2))
    
    # optimizer
    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    # TensorFlow session
    start_time = time.time()
    init = tf.global_variables_initializer()
    losses = []
    
    with tf.Session() as session:
        
        session.run(init)
        
        for epoch in range(epochs):
            avg_cost = 0
        
            for x_batch in X_split:
            
                _, eval_loss = session.run([optimize, loss],
                                           feed_dict={x: x_batch})
                
            avg_cost += eval_loss
        
            avg_cost /= num_batches
                
            losses.append(avg_cost)
            
            if epoch % 10 == 0: 
                print("epoch: {}".format(epoch))
                print("loss: {}".format(avg_cost))
        
        # Encoded Input
        X_encoded = session.run([hidden_2], feed_dict = {x: X})
    
    final_time = time.time()
    print("Autoencoder Training Time: {} minutes".format((final_time - start_time)/60))
    
    # loss 
    print("Loss plot:")
    plt.plot(losses)
    plt.show()

    
    ### logistic regression with encoded input
    print("Fitting logistic regression to encoded data...")
    
    y = y.reset_index()['loan_status']
    x_encoded = pd.DataFrame(X_encoded[0])
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2,
                                                        random_state = 4290)
    log_reg = LogisticRegression(C=1000)
    log_reg.fit(X_train, y_train)
    
    ### evaluation
    print("Evaluation...")
    # scores
    y_scores_train = pd.DataFrame(y_train.reset_index())
    y_scores_train["scores"] = pd.DataFrame(log_reg.predict_proba(X_train)).loc[:,1]
    y_scores_train.columns = ["id","loan_status","scores"]
    
    y_scores_test = pd.DataFrame(y_test.reset_index())
    y_scores_test["scores"] = pd.DataFrame(log_reg.predict_proba(X_test)).loc[:,1]
    y_scores_test.columns = ["id","loan_status","scores"]
    
    # writing scores
    y_scores_train.to_csv("../output/scores/y_scores_train_encoded_logit.csv", sep = "^", 
                          index = False)
    y_scores_test.to_csv("../output/scores/y_scores_test_encoded_logit.csv", sep = "^", 
                         index = False)
    
    metrics = model_evaluation(y_train, y_test, y_scores_train["scores"], y_scores_test["scores"])
    
    return log_reg, metrics