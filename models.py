from keras.layers import Dense, LeakyReLU, Activation
from keras.layers import Input, Activation, BatchNormalization, Lambda, Dropout, Concatenate, Reshape, Dot , Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from tqdm import tqdm
from utils import *
from functools import partial

class GAN(object):
    """
       Basic Generative Adversarial Network
    """

    def __init__(self, g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size=64,
                 mode='bayes_by_gaussian_dropout'):
        """
        :param mode: ['basic', 'deep_implicit', 'bayes_by_gaussian_dropout']
        """
        self.g_input_dim = g_input_dim
        self.g_num_hidden = g_num_hidden
        self.g_hidden_dim = g_hidden_dim
        self.g_out_dim = g_out_dim
        self.c_num_hidden = c_num_hidden
        self.c_hidden_dim = c_hidden_dim
        self.sample_size = sample_size
        self.mode = mode
        self.performance_log = {'critic': [], 'generator': []}
        self.build_model()
        # self.sample_real= partial( np.random.multivariate_normal, mean=np.zeros(shape=(self.g_out_dim,)), cov=np.eye(self.g_out_dim) )
        self.sample_real=self._sample_real


    @staticmethod
    def build_gen_basic(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim):
        z = Input(shape=(g_input_dim,))
        h = Dense(g_hidden_dim, kernel_regularizer=l1_l2(1e-5, 1e-5), name='G_h_1')(z)
        h = LeakyReLU()(h)
        # h = Activation('tanh')(h)
        for i in range(g_num_hidden - 2):
            h = Dense(g_hidden_dim, kernel_regularizer=l1_l2(1e-5, 1e-5), name='G_h_{}'.format(i + 2))(h)
            h = LeakyReLU()(h)
        # h = Activation('tanh')(h)
        x = Dense(g_out_dim, name='G_out')(h)

        return Model(inputs=z, outputs=x, name="G")

    @staticmethod
    def build_gen_dpimp(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim):
        z = Input(shape=(g_input_dim,))
        h = Dense(g_hidden_dim, kernel_regularizer=l1_l2(1e-5, 1e-5), name='G_h_1')(z)
        h = LeakyReLU()(h)
        # h = Activation('tanh')(h)

        for i in range(g_num_hidden - 2):

            h= Lambda(lambda x: K.concatenate( [x,K.random_normal(shape=K.shape(x), mean=0, stddev=1)]) )(h)
            h = Dense(g_hidden_dim, kernel_regularizer=l1_l2(1e-5, 1e-5), name='G_h_{}'.format(i + 2))(h)
            h = LeakyReLU()(h)
            # h = Activation('tanh')(h)
        h = Lambda(lambda x: K.concatenate( [x, K.random_normal(shape=K.shape(x), mean=0, stddev=1)]),)(h)
        x = Dense(g_out_dim, name='G_out')(h)

        return Model(inputs=z, outputs=x, name="G")

    @staticmethod
    def build_gen_bgdo(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim):
        epsilon_std = 0.1
        z = Input(shape=(g_input_dim,))
        h = Dense(g_hidden_dim, name='G_h_1'.format(1))(z)
        # h = Activation('tanh')(h)
        h = LeakyReLU()(h)

        for i in range(g_num_hidden - 2):
            h_mean = Dense(g_hidden_dim, name='G_h_{}'.format(i + 2))(h)
            h__log_var = Dense(g_hidden_dim, name='G_h_logvar{}'.format(i + 2))(h)
            h =Lambda(sampling, output_shape=(g_hidden_dim,), arguments={'dim': g_hidden_dim,'epsilon_std':epsilon_std}, name='G_h{}_st'.format(i + 2))([h_mean, h__log_var])
            # h = Activation('tanh')(h)
            h = LeakyReLU()(h)
        h_mean = Dense(g_out_dim, kernel_regularizer=l1_l2(1e-5, 1e-5), name='G_out_mean')(h)
        h__log_var = Dense(g_out_dim, kernel_regularizer=l1_l2(1e-5, 1e-5), name='G_out_logvar')(h)
        x = Lambda(sampling, output_shape=(g_out_dim,), arguments={'dim': g_out_dim, 'epsilon_std': epsilon_std},
                   name='G_out')([h_mean, h__log_var])
        # x = Dense(g_out_dim, name='G_out')(h)

        return Model(inputs=z, outputs=x, name="G")

    def build_generator(self,g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, mode):
        if mode == 'basic':
            return self.build_gen_basic(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim)
        elif mode == 'deep_implicit':
            return self.build_gen_dpimp(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim)
        elif mode == 'bayes_by_gaussian_dropout':
            return self.build_gen_bgdo(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim)


    @staticmethod
    def build_critic(c_input_dim, c_num_hidden, c_hidden_dim, mode):
        x = Input(shape=(c_input_dim,))
        dropout_rate = 0.1
        h = Dense(c_hidden_dim, kernel_initializer='he_normal', kernel_regularizer=l1_l2(1e-5, 1e-5), name='C_h_1')(x)
        h = LeakyReLU()(h)
        h = Dropout(dropout_rate)(h)

        for i in range(c_num_hidden - 1):
            h = Dense(c_hidden_dim, kernel_initializer='he_normal',  kernel_regularizer=l1_l2(1e-5, 1e-5),name='C_h_{}'.format(i + 2))(h)
            h = LeakyReLU()(h)
            h = Dropout(dropout_rate)(h)
        output = Dense(1, name='C_out')(h)
        output = Activation('sigmoid')(output)
        return Model(inputs=x, outputs=output, name="C")

    @staticmethod
    def c_optimizer():
        return Adam(0.01, beta_1=0.5, beta_2=0.9)

    @staticmethod
    def g_optmizer():
        return Adam(0.01, beta_1=0.5, beta_2=0.9)

    def prepair(self):
        # generator weights
        generator = self.build_generator(self.g_input_dim, self.g_num_hidden, self.g_hidden_dim, self.g_out_dim,
                                         self.mode)
        # generator.summary()
        # critic weights
        critic = self.build_critic(self.g_out_dim, self.c_num_hidden, self.c_hidden_dim, self.mode)
        # input to the generator
        #         z = Input(shape=(self.g_input_dim,))
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
        self.gan_model_tg.compile(optimizer=self.g_optmizer(), loss='binary_crossentropy')

    def build_critic_model(self):
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
        self.gan_model_tc.compile(optimizer=self.c_optimizer(), loss=['binary_crossentropy', 'binary_crossentropy'])

    def build_model(self):
        self.prepair()
        self.build_gan_model()
        self.build_critic_model()

    @staticmethod
    def get_real_labels(sample_size):
        return np.random.uniform(low=0.8, high=1.0, size=sample_size).reshape((sample_size, 1))

    @staticmethod
    def get_fake_labels(sample_size):
        return np.zeros((sample_size, 1), dtype=np.float32)

        # def sample_real(self,sample_size):
        #     return np.random.multivariate_normal(mean=np.zeros(shape=(self.g_out_dim,)), cov=np.eye(self.g_out_dim), size=sample_size )
    @staticmethod
    def _sample_real( size):
        a = np.linspace(0.0, 2 * np.pi, 9)
        a = a[:-1]
        r = 10
        centers_x = 1.0 + r * np.cos(a)
        centers_y = 1.0 + r * np.sin(a)
        num_samples = size // centers_x.shape[0]
        image_batch = np.zeros((size, 2))
        mean = np.array(list(zip(centers_x, centers_y)), dtype=np.float)

        for i in range(centers_x.shape[0]):
            if mean[i, 0] == mean[i, 1] == 1.0:
                continue
            image_batch[num_samples * i:num_samples * (i + 1)] = np.random.multivariate_normal(mean[i],
                                                                                               [[0.4, 0.0], [0, 0.4]],
                                                                                               num_samples)
        return image_batch

    def sample_fake(self, sample_size):

        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                          size=sample_size)
        return self.g_model.predict(z)

    def train_critic(self, n_steps=5):
        fake_labels = self.get_fake_labels(self.sample_size)
        for i in range(n_steps):
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
            real_samples = self.sample_real(size=self.sample_size)
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
                 mode='gan'):
        self.RandomWeightedAverage = partial(RandomWeightedAverage, sample_size=sample_size)
        super().__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size,
                         mode)

    @staticmethod
    def c_optimizer():
        return Adam(0.01, beta_1=0.5, beta_2=0.9)

    @staticmethod
    def g_optmizer():
        return Adam(0.01, beta_1=0.5, beta_2=0.9)

    @staticmethod
    def build_critic(c_input_dim, c_num_hidden, c_hidden_dim, mode):
        x = Input(shape=(c_input_dim,))
        # h = Dense(c_hidden_dim,  kernel_initializer='he_normal', name='C_h_1')(x)
        h = Dense(c_hidden_dim, kernel_regularizer=l1_l2(1e-4, 1e-4), kernel_initializer='he_normal', name='C_h_1')(x)
        h = LeakyReLU()(h)

        for i in range(c_num_hidden - 1):
            # h = Dense(c_hidden_dim,  kernel_initializer='he_normal',
            #           name='C_h_{}'.format(i + 2))(h)
            h = Dense(c_hidden_dim, kernel_regularizer=l1_l2(1e-4, 1e-4), kernel_initializer='he_normal',
                      name='C_h_{}'.format(i + 2))(h)
            h = LeakyReLU()(h)
            h = Dropout(0.2)(h)
        output = Dense(1, name='C_out')(h)
        return Model(inputs=x, outputs=output, name="C")

    def build_gan_model(self):
        ### model to train generator###
        self.gan_model_tg = Sequential([self.g_model, self.c_freezed])
        self.gan_model_tg.compile(optimizer=self.g_optmizer(), loss=Wasserestein_loss)

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
                                  loss=[Wasserestein_loss, Wasserestein_loss, partial_gp_loss])

    @staticmethod
    def get_real_labels(sample_size):
        return np.ones((sample_size, 1), dtype=np.float32)

    @staticmethod
    def get_fake_labels(sample_size):
        return -np.ones((sample_size, 1), dtype=np.float32)

    #     def sample_real(self,sample_size):
    #         return np.random.multivariate_normal(mean=np.zeros(shape=(self.g_out_dim,)), cov=np.eye(self.g_out_dim), size=sample_size )

    def train_critic(self, n_steps=5):
        fake_labels = self.get_fake_labels(self.sample_size)
        dummy_labels = np.zeros((self.sample_size, 1), dtype=np.float32)
        for i in range(n_steps):
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
            real_samples = self.sample_real(size=self.sample_size)
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


def build_main_net(w=None, x=None, config=None):

    units, activation, output_activation = config['units'], config['activation'], config['output_activation']

    start, stop = get_weight_indices(units)
    x= Input(shape=(None,units[0],))
    w = Input(shape=(weights_dim(units),))

    ONES = Lambda(col_one)(x)
    H = Concatenate()([x, ONES])
    W = Lambda(slice_weights,arguments={'start':start[0],'stop':stop[0]})(w)
    W = Reshape((units[0]+1, units[1]))(W)
    H = Dot(axes=[-1,-2],name='Dense_0')([H,W])
    H = Activation(activation)(H)

    for i in range(1,len(start)):
        ONES = Lambda(col_one)(H)
        H = Concatenate()([H, ONES])
        W = Lambda(slice_weights, arguments={'start':start[i],'stop':stop[i]} )(w)
        W = Reshape((units[i]+1, units[i+1]))(W)
        H = Dot(axes=[-1,-2],name='Dense_'+str(i))([H,W])
        if i< len(start)-1:
            H = Activation(activation)(H)
    outp = Activation(output_activation)(H)

    return Model(inputs=[x,w],outputs=[outp], name="M")


class Hyper_Net_GAN(GAN):

    def __init__(self, main_net_config, g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim,
                 sample_size=64, mode='gan'):
        self.main_net_config = main_net_config
        g_hidden_dim = g_out_dim = weights_dim(main_net_config['units'])

        super().__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size,
                         mode)

    def build_gan_model(self):
        ### model to train generator###
        x = Input(shape=(None, self.main_net_config['units'][0],), name='net_in')
        z_ = Input(shape=(self.g_input_dim,), name='g_in')
        w = self.g_model(z_)
        c_out = self.c_freezed(w)
        # main net weights
        main_net = build_main_net(config=self.main_net_config)
        y = main_net([x, w])
        self.gan_model_pred = Model(inputs=[z_, x], outputs=y)
        self.gan_model_tg = Model(inputs=[z_, x], outputs=[c_out, y])
        self.gan_model_tg.compile(optimizer=self.g_optmizer(),
                                  loss={'C_freezed': 'binary_crossentropy', 'M': loglik_normal},
                                  loss_weights={'C_freezed': 0.5, 'M': 0.5})

    #     def build_critic_model(self):
    #         #### models to train descriminator###
    #         # noise input
    #         z = Input(shape=(self.g_input_dim,))
    #         # fake samples
    #         fake = self.g_freezed(z)
    #         # critic output for fake samples
    #         c_out_fake = self.c_model(fake)
    #         # real sample
    #         real = Input(shape=(self.g_out_dim,))
    #         # critic output for real samples
    #         c_out_real =self.c_model(real)

    #         self.gan_model_tc =Model(inputs=[z, real], outputs = [c_out_fake, c_out_real ])
    #         self.gan_model_tc.compile(optimizer=self.c_optimizer(), loss=['binary_crossentropy', 'binary_crossentropy'])

    #     def build_model(self):
    #         super().build_model()

    def train_generator(self, x, y):
        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                          size=self.sample_size)
        self.performance_log['generator'].append(
            self.gan_model_tg.train_on_batch({'g_in': z, 'net_in': x}, [self.get_real_labels(self.sample_size), y]))

    def train(self, x, y, n_train=100, n_c_pretrain=100, n_c_train_perit=1, n_c_train_perinterval=10,
              c_train_interval=100):

        x = np.tile(x, (self.sample_size, 1)).reshape((self.sample_size, *x.shape))
        y = np.tile(y, (self.sample_size, 1)).reshape((self.sample_size, *y.shape))

        self.train_critic(n_c_pretrain)

        for i in tqdm(range(n_train)):
            if i % c_train_interval == 0:
                self.train_critic(n_c_train_perinterval)
            self.train_critic(n_c_train_perit)
            self.train_generator(x, y)

    def predict(self, x, z=None, sample_size=1):

        if z is None:
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
        else:
            sample_size = z.shape[0]

        X = np.tile(x, (sample_size, 1)).reshape((sample_size, *x.shape))

        return self.gan_model_pred.predict([z, X])


class Hyper_Net_WGAN(WGAN_gp):

    def __init__(self, main_net_config, g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim,
                 sample_size=64, mode='gan'):
        self.main_net_config = main_net_config
        g_hidden_dim = g_out_dim = weights_dim(main_net_config['units'])
        super().__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, sample_size,
                         mode)

    def build_gan_model(self):
        ### model to train generator###
        x = Input(shape=(None, self.main_net_config['units'][0],), name='net_in')
        z_ = Input(shape=(self.g_input_dim,), name='g_in')
        w = self.g_model(z_)
        c_out = self.c_freezed(w)
        # main net weights
        main_net = build_main_net(config=self.main_net_config)
        y = main_net([x, w])
        self.gan_model_pred = Model(inputs=[z_, x], outputs=y)
        self.gan_model_tg = Model(inputs=[z_, x], outputs=[c_out, y])
        self.gan_model_tg.compile(optimizer=self.g_optmizer(),
                                  loss={'C_freezed': Wasserestein_loss, 'M': loglik_normal},
                                  loss_weights={'C_freezed': 0.5, 'M': 0.5})

    #     def build_model(self):
    #         super().build_model()

    def train_generator(self, x, y):
        z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                          size=self.sample_size)
        self.performance_log['generator'].append(
            self.gan_model_tg.train_on_batch({'g_in': z, 'net_in': x}, [self.get_real_labels(self.sample_size), y]))

    def train(self, x, y, n_train=100, n_c_pretrain=100, n_c_train_perit=1, n_c_train_perinterval=10,
              c_train_interval=100):

        x = np.tile(x, (self.sample_size, 1)).reshape((self.sample_size, *x.shape))
        y = np.tile(y, (self.sample_size, 1)).reshape((self.sample_size, *y.shape))

        self.train_critic(n_c_pretrain)

        for i in tqdm(range(n_train)):
            if i % c_train_interval == 0:
                self.train_critic(n_c_train_perinterval)
            self.train_critic(n_c_train_perit)
            self.train_generator(x, y)

    def predict(self, x, z=None, sample_size=1):

        if z is None:
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
        else:
            sample_size = z.shape[0]

        X = np.tile(x, (sample_size, 1)).reshape((sample_size, *x.shape))

        return self.gan_model_pred.predict([z, X])

