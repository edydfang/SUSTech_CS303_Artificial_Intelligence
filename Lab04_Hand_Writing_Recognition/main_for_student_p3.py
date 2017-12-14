#!/usr/bin/env python3
import os
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import random, shuffle
# matplotlib.use('Agg')


def one_hot_encoding(labels, bound):
    result = np.zeros((labels.size, bound))
    result[np.arange(labels.size), labels] = 1
    return result


def relu(vector):
    '''
    ReLU function Max{0, vector} for hidden layer
    '''
    return np.maximum(vector, 0, vector)


def softmax(X):
    '''
    Softmax function for output layer
    '''
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def loss_and_acc(X, Y, w, b):
    N = X.shape[0]
    Y_hat, _, _ = forward_prop(X, w[0], w[1], w[2], b[0], b[1], b[2])
    #loss = -np.sum(Ys_true.T * np.log(Y_hats)) / N
    acc = (np.argmax(Y_hat, axis=1) ==
           np.argmax(Y, axis=1)).mean() * 100
    # return loss, acc
    print(acc)


def forward_prop(X, w1, w2, w3, b1, b2, b3):
    z2 = X.dot(w1) + b1
    a2 = relu(z2)
    z3 = a2.dot(w2) + b2
    a3 = relu(z3)
    z4 = a3.dot(w3) + b3
    Y_hat = softmax(z4)
    #print("a", z2,a2,a3)
    return Y_hat, [z2, z3, z4], [X, a2, a3]


def d_relu(X):
    # 0-1
    return np.where(X < 0, 0, X)


def back_prop(Y_hat, y_ture, w, b, a, z):
    #N = Y_hat.shape[0]
    N = 1
    delta3 = - (y_ture - Y_hat)
    delta2 = (w[2].dot(delta3.T)).T * d_relu(z[1])
    delta1 = (w[1].dot(delta2.T)).T * d_relu(z[0])

    g_w3 = a[2].T.dot(delta3) / N
    g_w2 = a[1].T.dot(delta2) / N
    g_w1 = a[0].T.dot(delta1) / N
    # print(a[2])
    # print(delta1.shape)
    g_b1 = np.sum(delta1, axis=0) / N
    g_b2 = np.sum(delta2, axis=0) / N
    g_b3 = np.sum(delta3, axis=0) / N

    return [g_w1, g_w2, g_w3], [g_b1, g_b2, g_b3]


# Definition of functions and parameters
'''
matrix format
input matrx: sample_num*dimension_size
weight matrix: input_dimension*output_dimension
'''
# for example
EPOCH = 100
alpha = 0.1

# Read all data from .pkl
(train_images, train_labels, test_images, test_labels) = pickle.load(
    open('./mnist_data/data.pkl', 'rb'), encoding='latin1')
train_images = np.array(train_images)
train_labels = one_hot_encoding(np.array(train_labels), 10)
test_images = np.array(test_images)
test_labels = one_hot_encoding(np.array(test_labels), 10)

# 1. Data preprocessing: normalize all pixels to [0,1) by dividing 256
train_images = train_images / 256
test_images = test_images / 256

# 2. Weight initialization: Xavier
n1 = train_images.shape[1]
n2 = 300
n3 = 100
n4 = 10
w1 = (np.random.random((n1, n2)) - 0.5) * np.sqrt(6 / (n1 + n2))
w2 = (np.random.random((n2, n3)) - 0.5) * np.sqrt(6 / (n2 + n3))
w3 = (np.random.random((n3, n4)) - 0.5) * np.sqrt(6 / (n3 + n4))
b1 = np.zeros(n2)
b2 = np.zeros(n3)
b3 = np.zeros(n4)

# 3. training of neural network
loss = np.zeros((EPOCH))
accuracy = np.zeros((EPOCH))


all_idx =range(train_images.shape[0])

for epoch, batch_idx in enumerate(np.array_split(all_idx, 100)):
    if epoch > 50:
        alpha = 0.01
    # Forward propagation
    Y_hat, z, a = forward_prop(train_images[batch_idx], w1, w2, w3, b1, b2, b3)
    # print(a[2])
    # Back propagation
    g_w, g_b = back_prop(Y_hat, train_labels[batch_idx], [
                         w1, w2, w3], [b1, b2, b3], a, z)

    # l2 = 0.001 # Regularization factor
    # Gradient update
    # print(np.max(g_w[0]))
    # break
    w1 -= alpha * g_w[0]
    w2 -= alpha * g_w[1]
    w3 -= alpha * g_w[2]
    b1 -= alpha * g_b[0]
    b2 -= alpha * g_b[1]
    b3 -= alpha * g_b[2]

    # After an epoch
    # Testing for accuracy
    w = [w1, w2, w3]
    b = [b1, b2, b3]
    loss_and_acc(train_images, train_labels, w, b)
    print("---test--")
    loss_and_acc(test_images, test_labels, w, b)
    #loss_on_val, acc_on_val = loss_and_acc(Xs_val, Ys_val, w1, w2, b1, b2)

# 4. Plot
# for example
# plt.figure(figsize=(12,5))
# ax1 = plt.subplot(111)
# ax1.plot(......)
# plt.xlabel(......)
# plt.ylabel(......)
# plt.grid()
# plt.tight_layout()
# plt.savefig('figure.pdf', dbi=300)
