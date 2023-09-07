# -*- coding: utf-8 -*-

#   *** Do not import any library except already imported libraries ***
import numpy as np
import math
import random
#   *** Do not import any library except already imported libraries ***

class AutoEncoder:
    def __init__(self, input_size: int, hidden_size: int, latent_size: int, output_size: int, learning_rate: float):
        '''
        Refer to mlp.py
        '''
        ############################################## EDIT HERE ###########################################
        # pintv = 0.01
        # nintv = -0.01
        # self.W_i1 = np.random.uniform(low = nintv, high = pintv, size = (hidden_size, input_size))
        # self.B_i1 = np.zeros((hidden_size, 1))
        # self.W_i2 = np.random.uniform(low = nintv, high = pintv, size = (latent_size, hidden_size))
        # self.B_i2 = np.zeros((latent_size, 1))
        
        # self.W_o1 = np.random.uniform(low = nintv, high = pintv, size = (hidden_size, latent_size))
        # self.B_o1 = np.zeros((hidden_size, 1))
        # self.W_o2 = np.random.uniform(low = nintv, high = pintv, size = (output_size, hidden_size))
        # self.B_o2 = np.zeros((output_size, 1))

        self.W_i1 = np.random.normal(loc=0.0, scale=np.sqrt(2 / hidden_size), size=(hidden_size, input_size))
        self.B_i1 = np.zeros((hidden_size, 1))
        self.W_i2 = np.random.normal(loc=0.0, scale=np.sqrt(2 / latent_size), size=(latent_size, hidden_size))
        self.B_i2 = np.zeros((latent_size, 1))
        
        self.W_o1 = np.random.normal(loc=0.0, scale=np.sqrt(2 / hidden_size), size=(hidden_size, latent_size))
        self.B_o1 = np.zeros((hidden_size, 1))
        self.W_o2 = np.random.normal(loc=0.0, scale=np.sqrt(2 / output_size), size=(output_size, hidden_size))
        self.B_o2 = np.zeros((output_size, 1))

        self.grad_W_i1 = np.zeros(self.W_i1.shape)
        self.grad_B_i1 = np.zeros(self.B_i1.shape)
        self.grad_W_i2 = np.zeros(self.W_i2.shape)
        self.grad_B_i2 = np.zeros(self.B_i2.shape)

        self.grad_W_o1 = np.zeros(self.W_o1.shape)
        self.grad_B_o1 = np.zeros(self.B_o1.shape)
        self.grad_W_o2 = np.zeros(self.W_o2.shape)
        self.grad_B_o2 = np.zeros(self.B_o2.shape)

        self.lr = learning_rate

        self.input = None
        self.latent = None
        self.hidden_i = None
        self.hidden_o = None
        self.output = None

        self.ReLU_mask_hi = None
        self.ReLU_mask_ho = None
        self.ReLU_mask_l = None
        '''
        Define any additional variables you need
        '''

        ################################################# END ##############################################

    def forward(self, x):
        ############################################## EDIT HERE ###########################################
        self.input = np.array(x).reshape((len(x), 1)) # reshape input to numpy array (input_size, 1)

        hidden_i = np.dot(self.W_i1, self.input) + self.B_i1
        self.hidden_i = self.ReLU(hidden_i)
        self.ReLU_mask_hi = self.ReLU_mask 

        latent = np.dot(self.W_i2, self.hidden_i) + self.B_i2
        self.latent = self.ReLU(latent)
        self.ReLU_mask_l = self.ReLU_mask 

        hidden_o = np.dot(self.W_o1, self.latent) + self.B_o1
        self.hidden_o = self.ReLU(hidden_o)
        self.ReLU_mask_ho = self.ReLU_mask 

        output = np.dot(self.W_o2, self.hidden_o) + self.B_o2
        self.output = output

        return self.output
        ################################################# END ##############################################

    def backward(self):
        ############################################## EDIT HERE ###########################################
        d_output = self.output
        d_output[self.output <= 0] = 0
        d_output[self.output > 0] = 1

        lr = self.lr
        d_output *= lr * (self.output - self.input)

        # Update weights and biases for the output layer
        grad_W_o2 = np.dot(d_output, self.hidden_o.T)
        grad_B_o2 = np.sum(d_output, axis=1, keepdims=True)

        # Calculate gradients for the hidden layer of the output
        d_hidden_o = np.dot(self.W_o2.T, d_output)
        d_hidden_o[self.hidden_o <= 0] = 0
        d_hidden_o[self.hidden_o > 0] = 1

        # Update weights and biases for the hidden layer of the output
        grad_W_o1 = np.dot(d_hidden_o, self.latent.T)
        grad_B_o1 = np.sum(d_hidden_o, axis=1, keepdims=True)

        # Calculate gradients for the latent layer
        d_latent = np.dot(self.W_o1.T, d_hidden_o)
        d_latent[self.latent <= 0] = 0
        d_latent[self.latent > 0] = 1

        # Update weights and biases for the latent layer
        grad_W_i2 = np.dot(d_latent, self.hidden_i.T)
        grad_B_i2 = np.sum(d_latent, axis=1, keepdims=True)

        # Calculate gradients for the hidden layer of the input
        d_hidden_i = np.dot(self.W_i2.T, d_latent)
        d_hidden_i[self.hidden_i <= 0] = 0
        d_hidden_i[self.hidden_i > 0] = 1

        # Update weights and biases for the hidden layer of the input
        grad_W_i1 = np.dot(d_hidden_i, self.input.T)
        grad_B_i1 = np.sum(d_hidden_i, axis=1, keepdims=True)
        ################################################# END ##############################################

    def step(self):
        self.W_i1 -= self.lr * self.grad_W_i1
        self.B_i1 -= self.lr * self.grad_B_i1
        self.W_i2 -= self.lr * self.grad_W_i2
        self.B_i2 -= self.lr * self.grad_B_i2
        
        self.W_o1 -= self.lr * self.grad_W_o1
        self.B_o1 -= self.lr * self.grad_B_o1
        self.W_o2 -= self.lr * self.grad_W_o2
        self.B_o2 -= self.lr * self.grad_B_o2

    def ReLU(self, x):
        self.ReLU_mask = np.zeros(x.shape)
        self.ReLU_mask[x >= 0] = 1.0

        return np.multiply(self.ReLU_mask, x)