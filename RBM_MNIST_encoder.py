import numpy as np
from keras.datasets import mnist
import random
import matplotlib.pyplot as plt

#Architecture
num_visible = 784
num_hidden = 100 

weights = np.random.randn(num_visible, num_hidden).astype(np.float32)
bias_hidden_c = np.random.randn(num_hidden).astype(np.float32)
bias_visible_b = np.random.randn(num_visible).astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample(activations):
    return (activations > np.random.rand(*activations.shape)).astype(np.float32)


def positive_phase(inputs, bias_vec_hidden):
    global weights
    
    product= np.dot(inputs,weights)+bias_vec_hidden
    activations_prop = sigmoid(product)
    samples_hid = sample(activations_prop)#Sample Hidden units 
    
    return activations_prop , samples_hid


def negative_phase(sample_hid, bias_vec_visible, bias_vec_hidden):  # Reconstruction
    global weights
    
    # Schritt 1: h0 → v1
    product = np.dot(sample_hid, weights.T) + bias_vec_visible
    activations_x0 = sigmoid(product)   # v1_prob

    samples_vis = sample(activations_x0)  # v1_sample

    # Schritt 2: v1 → h1
    product = np.dot(samples_vis, weights) + bias_vec_hidden   
    activations_x1 = sigmoid(product)   # h1_prob

    return activations_x0, samples_vis, activations_x1



def update_parameters(v0, h0_prob, v1_sample, h1_prob, learning_rate=0.05): #update parameter to maximze liklihood of Inputs paramters 
    global weights
    global bias_hidden_c
    global bias_visible_b
    
    weight_step = np.outer(v0, h0_prob) - np.outer(v1_sample, h1_prob)
    bias_vis_step = v0-v1_sample #v1 ist the reconstruction of the inputs by rbm
    bias_hid_step = h0_prob - h1_prob

    weights += learning_rate*weight_step
    bias_visible_b += learning_rate*bias_vis_step
    bias_hidden_c += learning_rate*bias_hid_step



def train(train_set, valid_set, num_of_epochs=100):
    global bias_hidden_c, bias_visible_b, weights

    mse_errors = []
    valid_errors = []

    best_valid_error = float("inf")

    for epoch in range(num_of_epochs):
        print("Epoche " + str(epoch))

        # ===== TRAINING =====
        squared_error = 0

        for i in range(len(train_set)):
            v0 = train_set[i]

            h0_prob, h0_sample = positive_phase(v0, bias_hidden_c)
            v1_prob, v1_sample, h1_prob = negative_phase(
                h0_sample, bias_visible_b, bias_hidden_c
            )

            update_parameters(v0, h0_prob, v1_sample, h1_prob)

            squared_error += np.sum((v0 - v1_prob) ** 2)

        train_error = squared_error / len(train_set)
        mse_errors.append(train_error)

        # ===== VALIDATION (OHNE UPDATE!) =====
        valid_err = 0

        for i in range(len(valid_set)):
            v0 = valid_set[i]

            h0_prob, h0_sample = positive_phase(v0, bias_hidden_c)
            v1_prob, _, _ = negative_phase(
                h0_sample, bias_visible_b, bias_hidden_c
            )

            valid_err += np.sum((v0 - v1_prob) ** 2)

        valid_err /= len(valid_set)
        valid_errors.append(valid_err)

        print(f"Train Error: {train_error:.4f}, Valid Error: {valid_err:.4f}")

        # ===== EARLY STOPPING =====
        '''if valid_err > best_valid_error:
            print("Early stopping ausgelöst!")
            break'''

        best_valid_error = valid_err

    return mse_errors, valid_errors


    

            





