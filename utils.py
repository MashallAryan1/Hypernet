from functools import partial
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, BatchNormalization, Activation,LeakyReLU, Lambda, Dropout, Concatenate, GaussianNoise
import keras
import numpy as np


def transfer(name, inputs):
    if name.lower()=='leakyrelu':
        return LeakyReLU()(inputs)
    else:
        return Activation(name)(inputs)

# gaussian layer with reparametrization trick
# epsilon_std =5.0001
def sampling(args, dim,epsilon_std):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon



def get_weight_indices(units):

    indices, w_counter = [], 0
    for i in range(len(units) - 1):
        input_shape, output_shape = units[i], units[i + 1]

        w_dim = input_shape * output_shape
        b_dim = output_shape

        indices.append((w_counter, w_counter + w_dim, w_counter + w_dim, w_counter + w_dim + b_dim))
        w_counter += w_dim + b_dim
    return indices


def slice_weights(w, start,stop):
    return w[:, start:stop]


def weights_dim(units):
    num_weights=0
    if isinstance(units, int):
        return units
    for i in range(len(units)-1):
        num_weights += (units[i]+1)* units[i+1]
    return num_weights


def wasserestein_real(y_true, y_pred):
    return K.mean(y_pred)

def wasserestein_fake(y_true, y_pred):
    return  -K.mean(y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def RandomWeightedAverage(inputs, sample_size):
    weights = K.random_uniform((sample_size, 1))
    return (weights * inputs[0]) + ((1 - weights) * inputs[1])




def binary_crossentropy_from_logit(y_true, y_pred):
    return K.mean( keras.losses.binary_crossentropy(y_true, K.sigmoid(y_pred)))

def generator_loss_logit(y_true, y_pred):
    return -K.mean( y_pred, axis=-1)

def hing_real(y_true, y_pred):
    return K.mean( K.maximum(K.zeros_like(y_true),K.ones_like(y_pred)- y_pred))
def hing_fake(y_true, y_pred):
    return K.mean( K.maximum(K.zeros_like(y_true),K.ones_like(y_pred)+ y_pred))

def square_hing_real(y_true, y_pred):
    return K.mean(K.square( K.maximum(K.zeros_like(y_true),K.ones_like(y_pred)- y_pred)))
def square_hing_fake(y_true, y_pred):
    return K.mean(K.square( K.maximum(K.zeros_like(y_true),K.ones_like(y_pred)+ y_pred)))



def main_net_neg_loglikelihood(y_true, y_pred):
    # return -K.mean(dist.log_prob(y_pred- y_true ))
    return K.mean(K.square(y_pred - y_true))
