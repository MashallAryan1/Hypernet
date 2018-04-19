from functools import partial
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, BatchNormalization, Activation,LeakyReLU, Lambda, Dropout
import numpy as np


class Hidden(object):
    def __init__(self, ltype, activation='leakyrelu',  kernel_regularizer= None,batch_norm=True, dropout_rate=0.0):
        self.ltype = ltype
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate= dropout_rate
        self.kernel_regularizer =  kernel_regularizer

    def __call__(self, inputs, dims, name_prefix= 'Layer', **kwargs):

        if self.ltype == 'basic':
            return self.basic(inputs, dims, name_prefix)
        elif  self.ltype == 'gaussian':
            return self.gaussian(inputs, dims, name_prefix, epsilon_stddev= kwargs.get('epsilon_stddev', 1.0))
        elif self.ltype == 'deep_st':
            return self.deep_stochastic(inputs, dims, name_prefix, epsilon_stddev= kwargs.get('epsilon_stddev', 1.0))
        elif self.ltype == 'deepst_n_gaussian':
            return self.deepst_n_gaussian(inputs, dims, name_prefix, epsilon_stddev= kwargs.get('epsilon_stddev', 1.0))

    def basic(self, inputs, dims, name_prefix):
        for i, dim in enumerate(dims):
            if i == 0:
                h = Dense(dim,   kernel_regularizer=self.kernel_regularizer(),name = name_prefix+'_{}'.format(i))(inputs)
                if self.batch_norm:
                    h = BatchNormalization()(h)
                h = transfer(self.activation, h)
                if self.dropout_rate > 0:
                    h = Dropout(self.dropout_rate)(h)
                continue
            h = Dense(dim, kernel_regularizer=self.kernel_regularizer(),name = name_prefix+'_{}'.format(i) )(h)
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = transfer(self.activation, h)
            if self.dropout_rate > 0:
                h = Dropout(self.dropout_rate)(h)
        return h

    def gaussian(self,inputs, dims, name_prefix, epsilon_stddev=1.0):
        for i, dim in enumerate(dims):
            if i == 0:
                h_mean = Dense(dim, kernel_regularizer=self.kernel_regularizer(),name = name_prefix+'_mean_{}'.format(i))(inputs)
                h_log_var = Dense(dim, name = name_prefix+'_logvar_{}'.format(i))(inputs)
                h = Lambda(sampling, output_shape=(dim,), arguments={'dim': dim,'epsilon_std':epsilon_stddev}, name=name_prefix+'_sampled_{}'.format(i))([h_mean, h_log_var])
                if self.batch_norm:
                    h = BatchNormalization()(h)
                h = transfer(self.activation, h)
                continue
            h_mean = Dense(dim, kernel_regularizer=self.kernel_regularizer(), name=name_prefix + '_mean_{}'.format(i))(h)
            h_log_var = Dense(dim, name=name_prefix + '_logvar_{}'.format(i))(h)
            h = Lambda(sampling, output_shape=(dim,), arguments={'dim': dim, 'epsilon_std': epsilon_stddev},
                       name=name_prefix + '_sampled_{}'.format(i))([h_mean, h_log_var])
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = transfer(self.activation, h)
        return h

    def deep_stochastic(self, inputs, dims, name_prefix, epsilon_stddev=1.0):
        for i, dim in enumerate(dims):
            if i == 0:
                h = Dense(dim, kernel_regularizer=self.kernel_regularizer(),name = name_prefix+'_{}'.format(i))(inputs)
                if self.batch_norm:
                    h = BatchNormalization()(h)
                h = transfer(self.activation, h)
                continue
            h = Lambda(lambda x: K.concatenate([x, K.random_normal(shape=K.shape(x), mean=0, stddev=epsilon_stddev)]))(h)
            h = Dense(dim, kernel_regularizer=self.kernel_regularizer(),name = name_prefix+'_{}'.format(i) )(h)
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = transfer(self.activation, h)
        return h

    def deepst_n_gaussian(self,inputs, dims, name_prefix='Layer', epsilon_stddev=1.0):
        for i, dim in enumerate(dims):
            if i == 0:
                h_mean = Dense(dim, name = name_prefix+'_mean_{}'.format(i))(inputs)
                h_log_var = Dense(dim, name = name_prefix+'_logvar_{}'.format(i))(inputs)
                h = Lambda(sampling, output_shape=(dim,), arguments={'dim': dim,'epsilon_std':epsilon_stddev}, name=name_prefix+'_sampled_{}'.format(i))([h_mean, h_log_var])
                h = Lambda(lambda x: K.concatenate([x, K.random_normal(shape=K.shape(x), mean=0, stddev=epsilon_stddev)]))(h)
                if self.batch_norm:
                    h = BatchNormalization()(h)
                h = transfer(self.activation, h)
                continue
            h_mean = Dense(dim, name=name_prefix + '_mean_{}'.format(i))(h)
            h_log_var = Dense(dim, name=name_prefix + '_logvar_{}'.format(i))(h)
            h = Lambda(sampling, output_shape=(dim,), arguments={'dim': dim, 'epsilon_std': epsilon_stddev},
                       name=name_prefix + '_sampled_{}'.format(i))([h_mean, h_log_var])
            h = Lambda(lambda x: K.concatenate([x, K.random_normal(shape=K.shape(x), mean=0, stddev=epsilon_stddev)]))(h)
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = transfer(self.activation, h)
        return h



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



def log_likelihood(y_true, y_pred, dist):
    return - dist.log_prob(y_pred - y_true)
loglik_normal = partial(log_likelihood, dist=tf.distributions.Normal(loc=0.0, scale=0.0001))
loglik_normal.__name__= 'loglik_normal'


def get_weight_indices(units):
    indices, size_acc  = [], 0
    for i in range(len(units)-1):
        input_shape,output_shape =units[i]+1, units[i+1]
        size = input_shape*output_shape
        indices.append((size_acc,size_acc+size))
        size_acc+=size
    return list(zip(*indices))


def slice_weights(w, start,stop):
    return w[:, start:stop]


def weights_dim(units):
    num_weights=0
    if isinstance(units, int):
        return units
    for i in range(len(units)-1):
        num_weights += (units[i]+1)* units[i+1]
    return num_weights


def col_one(x):
    return K.ones(shape= [K.shape(x)[i] for i in range(K.ndim(x)-1)]+[1])

def Wasserestein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

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

