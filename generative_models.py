import tensorflow as tf
from keras.layers import LeakyReLU, Activation, BatchNormalization, Input, Dense, Dropout, Concatenate, Lambda
from keras.models import Sequential, Model
import keras.backend as K
import numpy as np
from tqdm import tqdm
from  utils import *



LAYER_TYPES = ('DENSE', 'DEEP_IMPLICIT',)


class Network(object):
    def __init__(self, h_type='DENSE', activation='leakyrelu', output_activation=None, kernel_regularizer=None,
                 batch_norm=True,
                 dropout_rate=0.0):
        self.h_layer_type = h_type
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.output_activation = output_activation

    def fully_connected(self, dims, name_prefix):

        inputs = Input(shape=(dims[0],))
        for i, dim in enumerate(dims[1:-1]):
            # first layer
            if i == 0:
                h = Dense(dim, kernel_regularizer=self.kernel_regularizer(),
                          name=name_prefix + '_{}'.format(i))(inputs)
                if self.batch_norm:
                    h = BatchNormalization(name=name_prefix + '_BtNorm_{}'.format(i))(h)
                h = transfer(self.activation, h)
                if self.dropout_rate > 0:
                    h = Dropout(self.dropout_rate)(h)
                continue

            # hidden layers
            h = Dense(dim,  kernel_regularizer=self.kernel_regularizer(),
                      name=name_prefix + '_{}'.format(i))(h)
            if self.batch_norm:
                h = BatchNormalization(name=name_prefix + '_BtNorm_{}'.format(i))(h)
            h = transfer(self.activation, h)
            if self.dropout_rate > 0:
                h = Dropout(self.dropout_rate)(h)
        # output layer
        out = Dense(dims[-1],  kernel_regularizer=self.kernel_regularizer(),
                    name=name_prefix + '_{}'.format(len(dims)))(h)
        if  self.output_activation is not None:
            out = transfer(self.output_activation, out)
        return Model(inputs=inputs, outputs=out)

    def deep_implicit(self, dims, name_prefix, epsilon_stddev=1.0,):
        rand = True
        inputs = Input(shape=(dims[0],))
        for i, dim in enumerate(dims[1:-1]):
            rand = not rand
            if i == 0:
                h = Dense(dim,
                 kernel_regularizer=self.kernel_regularizer(), name=name_prefix + '_{}'.format(i))(inputs)
                if self.batch_norm:
                    h = BatchNormalization()(h)
                h = transfer(self.activation, h)
                continue
            if rand:
                h = Lambda(lambda x: K.concatenate([x, K.random_normal(shape=K.shape(x), mean=0, stddev=epsilon_stddev)]))(h)
            h = Dense(dim,
                 kernel_regularizer=self.kernel_regularizer(), name=name_prefix + '_{}'.format(i))(h)
            if self.batch_norm:
                h = BatchNormalization()(h)
            h = transfer(self.activation, h)
        # last layer
        if not rand:
            h = Lambda(lambda x: K.concatenate([x, K.random_normal(shape=K.shape(x), mean=0, stddev=epsilon_stddev)]))(h)
        out = Dense(dims[-1], kernel_regularizer=self.kernel_regularizer(),
                    name=name_prefix + '_{}'.format(len(dims)))(h)

        if self.output_activation is not None:
            out = transfer(self.output_activation, out)

        return Model(inputs=inputs, outputs=out)

    def __call__(self, dims, name_prefix='Layer', **kwargs):
        lt = self.h_layer_type.upper()

        if lt == 'DENSE':
            return self.fully_connected(dims, name_prefix)
        elif lt == 'DEEP_IMPLICIT':
            return self.deep_implicit(dims, name_prefix, epsilon_stddev=kwargs.get('epsilon_stddev', 1.0))


class GAN(object):
    """
       Basic Generative Adversarial Network
    """

    def __init__(self, g_config, c_config, noise_sample_size=64, real_samples=None, fake_label=0, real_label=1):
        """
        :param g_config : (dictionary) Containing the configuration of the generator, keys:
                          "type" : (string) from the list of LAYER_TYPES
                          "dims" : structure of the genarator e.g. [2,3,2] is a generator with 2d input a hidden layer with 3 hidden units and 2 output units
                          "activation": (string) activation for the hidden units
                          "output_activation":(string or None) activation function of the output layer
                          "batch_norm": (boolean) indicate the presence/absence of batch normalization layer
                          "optimizer" ": optimization algorithm for the generator
        :param c_config : (dictionary) Containing the configuration of the critic (similar to the g_config)
        :param noise_sample_size :(int) number of noise samples in the input of the generator

        """
        self.g_config = g_config
        self.c_config = c_config
        assert (self.c_config['dims'][0] == self.g_config['dims'][-1])
        self.noise_sample_size = noise_sample_size
        self.real_samples = real_samples
        if self.real_samples is None:
            self.real_samples = K.random_normal(shape=(self.noise_sample_size, self.c_config["dims"][0]), mean=0.0, stddev=1.0)

        self.init_model()
        # set the distribution/model that generates the real samples for the critic. (prior)
        self.performance_log = {'critic': [], 'generator': []}
        self.fake_label, self.real_label = fake_label,real_label


    def build_generator(self):
        """
        Create an generator
        :return: a generator z-x model of type LAYER_TYPES[self.mode]
        """
        cnet = Network(h_type=self.g_config['h_type'],
                       activation=self.g_config['activation'],
                       output_activation=self.g_config['output_activation'],
                       kernel_regularizer=self.g_config['kernel_regularizer'],
                       batch_norm=self.g_config['batch_norm'],
                       dropout_rate=self.g_config['dropout_rate'])
        # generator weights
        if self.g_config['h_type'] == 'DEEP_IMPLICIT' :
            generator = cnet(dims=self.g_config['dims'],sample_size=self.noise_sample_size, name_prefix='G')
        else:
            generator = cnet(dims=self.g_config['dims'], name_prefix='G')
        self.g_model = Sequential([generator], name="G_trainable")
        # non-trainable generator
        self.g_freezed = Sequential([generator], name="G_freezed")
        self.g_freezed.trainable = False


    def build_critic(self):
        cnet = Network(h_type=self.c_config['h_type'],
                       activation=self.c_config['activation'],
                       output_activation=self.c_config['output_activation'],
                       kernel_regularizer=self.c_config['kernel_regularizer'],
                       batch_norm=self.c_config['batch_norm'],
                       dropout_rate=self.c_config['dropout_rate'])
        # critic weights
        critic = cnet(dims=self.c_config['dims'], name_prefix='C')
        # trainable critic
        self.c_model = Sequential([critic], name="C_trainable")
        # non-trainable critic
        self.c_freezed = Sequential([critic], name="C_freezed")
        self.c_freezed.trainable = False

    def noise(self):
        return K.random_normal((self.noise_sample_size, self.g_config['dims'][0]))

    def init_model(self):
        """
        create  generator, discriminator and their non-trainable versions
        Put everything together to build the GAN
        :return: generator self.g_model, discriminator  self.c_model, non-trainable generator self.g_freezed,
                non-trainable discriminator
        """
        # noise input
        z = self.noise()

        self.build_generator()
        self.build_critic()

        # create a model to train generator
        self.build_gan_model(z)

        self.compile_g()
        # create a model to train the critic
        self.build_critic_model(z,self.real_samples)
        self.compile_c()


    def build_gan_model(self,z_tensor):
        """
         wire up generator and non-trainable discriminator/critic to  train generator
        :return: trainable GAN model self.gan_model_tg
        """
        ### model to train generator###
        z =  Input(shape=(self.g_config["dims"][0],),tensor=z_tensor)
        g_of_z = self.g_model(z)
        c_out = self.c_freezed(g_of_z)
        self.gan_model_tg = Model(inputs=z,outputs=c_out)



    def build_critic_model(self,z_tensor,real_tensor):
        """
        set up a model to train the discriminator
        :return: self.gan_model_tc
        """
        #### models to train descriminator###
        # noise input

        # fake samples
        z =  Input(shape=(self.g_config["dims"][0],),tensor=z_tensor)
        self.fake = self.g_freezed(z)
        # critic output for fake samples
        c_out_fake = self.c_model(self.fake)
        # real sample
        real = Input(shape=(self.g_config["dims"][-1],), tensor=K.cast(real_tensor,dtype="float32"))
        # critic output for real samples
        c_out_real = self.c_model(real)
        self.gan_model_tc = Model(inputs=[z, real], outputs=[c_out_fake, c_out_real])


    def compile_c(self):
        self.gan_model_tc.compile(optimizer=self.c_config['optimizer'](), loss=[self.c_config['loss']['fake'] , self.c_config['loss']['real']])

    def compile_g(self):
        self.gan_model_tg.compile(optimizer=self.g_config['optimizer'](), loss=self.g_config['loss'])



    def pre_train(self,n_pretrain):
        fake_labels =  self.get_fake_labels(self.noise_sample_size)
        real_labels =  self.get_real_labels(self.noise_sample_size)
        for j in range(n_pretrain):
                self.gan_model_tc.train_on_batch(x=None, y=[fake_labels, real_labels])

    def train(self, n_train, n_c_train,n_pretrain=0):
        fake_labels = self.get_fake_labels(self.noise_sample_size)
        real_labels = self.get_real_labels(self.noise_sample_size)

        for j in range(n_pretrain):
            self.performance_log['critic'].append(
                self.gan_model_tc.train_on_batch(x=None, y=[fake_labels, real_labels]))


        for i in tqdm(range(n_train)):
            for j in range(n_c_train):
                self.performance_log['critic'].append(self.gan_model_tc.train_on_batch(x=None,y=[fake_labels, real_labels]))
            self.performance_log['generator'].append(self.gan_model_tg.train_on_batch(x=None,y=real_labels))

    def sample_fake(self, sample_size,noise_var=1):
        """
        Samples from the generator
        :param sample_size: number of generated samples

        """
        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_config['dims'][0],)), cov=noise_var*np.eye(self.g_config['dims'][0],),
                                          size=sample_size)
        return self.g_model.predict(z)

    def get_loss(self, x):
        """
        :param x: a set of real or fake samples
        :return: loss value generated by the discriminator
        """
        return self.c_model.predict(x)


    @staticmethod
    def get_real_labels(sample_size):
        """
        Generates lables for real samples via one-sided label smoothing
        :param sample_size: (int) number of real samples
        :return: lables for real samples
        """
        return np.ones((sample_size, 1), dtype=np.float32)

        # return np.random.uniform(low=0.8, high=1.2, size=sample_size).reshape((sample_size, 1))

    @staticmethod
    def get_fake_labels(sample_size):
        """
        Generates labels for fake samples
        :param sample_size: (int) number of fake samples
        :return: lables for fake samples
        """
        return np.zeros((sample_size, 1), dtype=np.float32)


