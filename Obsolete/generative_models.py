from keras.layers import Dense, LeakyReLU, Activation
from keras.layers import Input, Activation, BatchNormalization, Lambda, Dropout, Concatenate, Reshape, Dot , Concatenate, Add
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l1_l2
from keras.losses import binary_crossentropy, mean_squared_error
from tqdm import tqdm
from utils import *
from functools import partial

#
# LAYER_TYPES = {'basic': 'basic', 'deep_implicit': 'deep_st', 'bayes_by_gaussian_dropout': 'gaussian',
#               'deepst_n_gaussian': 'deepst_n_gaussian'}


class GAN(object):
    """
       Basic Generative Adversarial Network
    """
    def __init__(self, g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size=64,
                 mode='bayes_by_gaussian_dropout'):
        """
        :param g_input_dim : (int) input dimensionality of the generator
        :param g_num_hidden : (int) number of hidden layers for the generator
        :param g_hidden_dim : (int) number of hidden units for the hidden layers of the generator
        :param c_num_hidden : (int) number of hidden layers for the critic/discriminator
        :param c_hidden_dim : (int) number of hidden units for the hidden layers of critic
        :param sample_size :(int) number of noise samples in the input of the generator
        :param mode: (str from LAYER_TYPES) generator type

        """
        self.g_input_dim = g_input_dim
        self.g_num_hidden = g_num_hidden
        self.g_hidden_dim = g_hidden_dim
        self.g_out_dim = g_out_dim
        self.c_num_hidden = c_num_hidden
        self.c_hidden_dim = c_hidden_dim
        self.sample_size = sample_size
        self.mode = mode
        # self.g_optmizer = self.c_optimizer = (lambda : Adam(0.002, beta_1=0.5, beta_2=0.9))
        # set the optimizers for generator and descriminator
        self.c_optimizer = (lambda : Adam())#0.001, beta_1=0.5, beta_2=0.99))
        self.g_optimizer = (lambda : Adam())#0.001, beta_1=0.5, beta_2=0.9))
        # set the distribution/model that generates the real samples for the critic. (prior)
        self.sample_real = (lambda sample_size: np.random.multivariate_normal(mean=np.zeros(shape=(self.g_out_dim,)), cov=np.eye(self.g_out_dim), size=sample_size ))
        self.performance_log = {'critic': [], 'generator': []}
        # build and initialize the generator and critic networks
        self.build_model()



    def build_generator(self):
        """
        Create an generator
        :return: a generator z-x model of type LAYER_TYPES[self.mode]
        """
        z = Input(shape=(self.g_input_dim,))
        h = Hidden(ltype=LAYER_TYPES[self.mode], kernel_regularizer =(lambda : l1_l2(1e-5, 1e-5)), batch_norm=True)(inputs=z, dims=[self.g_hidden_dim] * self.g_num_hidden, name_prefix ='G_h')
        x = Dense(self.g_out_dim, name='G_out')(h)

        return Model(inputs=z, outputs=x, name="G")


    def build_critic(self,):
        """
        Create a discriminator
        :return: discriminator model x->label
        """
        x = Input(shape=(self.g_out_dim,))
        dropout_rate = 0.1
        h = Hidden(ltype=LAYER_TYPES['dense'], kernel_regularizer =(lambda : l1_l2(1e-5, 1e-5)), batch_norm=False, dropout_rate=dropout_rate)(inputs=x, dims=[self.c_hidden_dim] * self.c_num_hidden, name_prefix ='C_h')
        output = Dense(1, name='C_out')(h)
        # output = Activation('sigmoid')(output)
        return Model(inputs=x, outputs=output, name="C")


    def prepair(self):
        """
        create  generator, discriminator and their non-trainable versions
        :return: generator self.g_model, discriminator  self.c_model, non-trainable generator self.g_freezed,
                non-trainable discriminator
        """

        # generator weights
        generator = self.build_generator()
        # generator.summary()
        # critic weights
        critic = self.build_critic()
        # generator model
        self.g_model = Sequential([generator])
        # trainable critic
        self.c_model = Sequential([critic])
        # non-trainable critic
        self.c_freezed = Sequential([critic], name="C_freezed")
        self.c_freezed.trainable = False
        # non-trainable generator
        self.g_freezed = Sequential([generator])
        self.g_freezed.trainable = False

    def build_gan_model(self):
        ### model to train generator###
        self.gan_model_tg = Sequential([self.g_model, self.c_freezed])
        self.gan_model_tg.compile(optimizer=self.g_optimizer(), loss=generator_loss_logit)

    def build_critic_model(self):
        """
        set up a model to train the discriminator
        :return: self.gan_model_tc
        """
        #### models to train descriminator###
        # noise input
        z = Input(shape=(self.g_input_dim,))

        # fake samples
        fake = self.g_freezed(z)
        # critic output for fake samples
        c_out_fake = self.c_model(fake)
        # real sample
        real = Input(shape=(self.g_out_dim,))
        # critic output for real samples
        c_out_real = self.c_model(real)

        self.gan_model_tc = Model(inputs=[z, real], outputs=[c_out_fake, c_out_real])
        self.gan_model_tc.compile(optimizer=self.c_optimizer(), loss=[binary_crossentropy_from_logit,binary_crossentropy_from_logit])

    def build_model(self):
        """
        Put everything together to build the GAN

        """
        # create generator and discriminator models
        self.prepair()
        # create a model to train generator
        self.build_gan_model()
        # create a model to train the critic
        self.build_critic_model()


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


    def sample_fake(self, sample_size):
        """
        Samples from the generator
        :param sample_size: number of generated samples

        """
        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                          size=sample_size)
        return self.g_model.predict(z)

    def train_critic(self, n_steps=5):
        """
        Train the discriminator
        :param n_steps: number of training steps
        """
        # produce lables for the fake samples
        fake_labels = self.get_fake_labels(self.sample_size)
        for i in range(n_steps):
            #
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
            real_samples = self.sample_real(self.sample_size)
            real_labels = self.get_real_labels(self.sample_size)
            self.performance_log['critic'].append(
                self.gan_model_tc.train_on_batch([z, real_samples], [fake_labels, real_labels]))

    def train_generator(self):
        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                          size=self.sample_size)
        self.performance_log['generator'].append(
            self.gan_model_tg.train_on_batch(z, np.ones(shape=(self.sample_size, 1))))

    def train(self, n_train=100, n_c_pretrain=100, n_c_train_perit=1, n_c_train_perinterval=0, c_train_interval=100):
        self.train_critic(n_c_pretrain)
        for i in tqdm(range(n_train)):
            if i % c_train_interval == 0:
                self.train_critic(n_c_train_perinterval)
            self.train_critic(n_c_train_perit)
            self.train_generator()


class WGAN_gp(GAN):

    def __init__(self, g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size=64,
                 mode='bayes_by_gaussian_dropout'):
        self.RandomWeightedAverage = partial(RandomWeightedAverage, sample_size=sample_size)
        super().__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size,
                         mode)


    def build_critic(self,):
        dropout_rate = 0.1
        x = Input(shape=(self.g_out_dim,))
        h = Hidden(ltype=LAYER_TYPES['dense'], kernel_regularizer =(lambda : l1_l2(1e-5, 1e-5)), batch_norm=False, dropout_rate=dropout_rate)(inputs=x, dims=[self.c_hidden_dim] * self.c_num_hidden, name_prefix ='C_h')
        output = Dense(1, name='C_out')(h)
        return Model(inputs=x, outputs=output, name="C")

    def build_gan_model(self):
        ### model to train generator###
        self.gan_model_tg = Sequential([self.g_model, self.c_freezed])
        self.gan_model_tg.compile(optimizer=self.g_optimizer(), loss=generator_loss_logit)

    def build_critic_model(self):
        # noise input
        z = Input(shape=(self.g_input_dim,))
        # fake samples
        fake = self.g_freezed(z)
        # critic output for fake samples
        c_out_fake = self.c_model(fake)
        # real sample
        real = Input(shape=(self.g_out_dim,))
        # critic output for real samples
        c_out_real = self.c_model(real)

        averaged_samples = Lambda(self.RandomWeightedAverage)([real, fake])
        # We then run these samples through the discriminator as well. Note that we never really use the discriminator
        # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
        averaged_samples_out = self.c_model(averaged_samples)

        # The gradient penalty loss function requires the input averaged samples to get gradients. However,
        # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
        # of the function with the averaged samples here.
        GRADIENT_PENALTY_WEIGHT = 10
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

        self.gan_model_tc = Model(inputs=[z, real], outputs=[c_out_fake, c_out_real, averaged_samples_out])
        self.gan_model_tc.compile(optimizer=self.c_optimizer(),
                                  loss=[wasserestein_fake, wasserestein_real, partial_gp_loss])

    @staticmethod
    def get_real_labels(sample_size):
        return np.ones((sample_size, 1), dtype=np.float32)

    @staticmethod
    def get_fake_labels(sample_size):
        return np.zeros((sample_size, 1), dtype=np.float32)#np.ones((sample_size, 1), dtype=np.float32)

    #     def sample_real(self,sample_size):
    #         return np.random.multivariate_normal(mean=np.zeros(shape=(self.g_out_dim,)), cov=np.eye(self.g_out_dim), size=sample_size )

    def train_critic(self, n_steps=5):
        fake_labels = self.get_fake_labels(self.sample_size)
        dummy_labels = np.zeros((self.sample_size, 1), dtype=np.float32)
        for i in range(n_steps):
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
            real_samples = self.sample_real(self.sample_size)
            real_labels = self.get_real_labels(self.sample_size)
            self.performance_log['critic'].append(
                self.gan_model_tc.train_on_batch([z, real_samples], [fake_labels, real_labels, dummy_labels]))


    def train_generator(self):
        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                          size=self.sample_size)
        self.performance_log['generator'].append(
            self.gan_model_tg.train_on_batch(z, self.get_real_labels(self.sample_size)))


    def train(self, n_train=100, n_c_pretrain=100, n_c_train_perit=1, n_c_train_perinterval=10, c_train_interval=100):
        self.train_critic(n_c_pretrain)
        for i in tqdm(range(n_train)):
            if i % c_train_interval == 0:
                self.train_critic(n_c_train_perinterval)
            self.train_critic(n_c_train_perit)
            self.train_generator()






class  EEGAN(GAN):

    def __init__(self,g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, e_num_hidden,
                 e_hidden_dim, sample_size=64,  mode='bayes_by_gaussian_dropout'):
        super(EEGAN, self).__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden,
                                    c_hidden_dim, sample_size,mode)

        self.e_num_hidden, self.e_hidden_dim = e_num_hidden, e_hidden_dim

    def encoder(self):
        x = Input(shape=(self.g_out_dim,))
        h= Hidden(ltype=LAYER_TYPES[self.mode], kernel_regularizer=(lambda: l1_l2(1e-5, 1e-5)), batch_norm=True)(inputs=x, dims=[self.e_hidden_dim] * self.e_num_hidden, name_prefix='E_h')
        z = Dense(self.g_out_dim, name='e_out')(h)
        return Model(inputs=x,outputs=z)


    def build_gan_model(self):
        ### model to train generator###
        encoder = self.encoder()
        ### model to train generator###
        z = Input(shape=(self.g_input_dim,), name='g_in')
        x = self.g_model(z)
        z_prime = encoder(x)

        c_out = self.c_freezed(x)
        self.gan_model_tg = Model(inputs=[z], outputs=[c_out])

        def eegan_loss(y_true,y_pred):
                prior = binary_crossentropy(y_true,y_pred)
                reconstruction = 0.005* mean_squared_error(z,z_prime)
                return prior+reconstruction

        self.gan_model_tg.compile(optimizer=self.g_optimizer(), loss=eegan_loss)

