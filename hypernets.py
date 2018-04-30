from generative_models import *
from keras.layers import Dense, LeakyReLU, Activation
from keras.layers import Input, Activation, BatchNormalization, Lambda, Dropout, Concatenate, Reshape, Dot , Concatenate, Add
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.losses import binary_crossentropy, mean_squared_error
from tqdm import tqdm
from utils import *
from functools import partial


def build_main_net(w=None, x=None, config=None):
    units, activation, output_activation = config['units'], config['activation'], config['output_activation']

    indices = get_weight_indices(units)
    start_w, stop_w, start_b, stop_b = indices[0]
    x = Input(shape=(None, units[0],), name="X")
    w = Input(shape=(weights_dim(units),), name="W_raw")
    #     X = Lambda((lambda x: K.tile(x,sample_size)),name="X_tiled" )(x)

    W = Lambda(slice_weights, arguments={'start': start_w, 'stop': stop_w}, name='MainW_0_sliced')(w)
    W = Reshape((units[0], units[1]), name='MainW_0')(W)
    B = Lambda(slice_weights, arguments={'start': start_b, 'stop': stop_b}, name='MainB_0')(w)

    H = Dot(axes=[-1, -2])([x, W])
    H = Add(name='Dense_0')([H, B])
    H = Activation(activation)(H)

    for i in range(1, len(indices)):
        start_w, stop_w, start_b, stop_b = indices[i]
        W = Lambda(slice_weights, arguments={'start': start_w, 'stop': stop_w}, name='MainW_{}_sliced'.format(i))(w)
        W = Reshape((units[i], units[i + 1]), name='MainW_{}'.format(i))(W)
        B = Lambda(slice_weights, arguments={'start': start_b, 'stop': stop_b}, name='MainB_{}'.format(i))(w)

        H = Dot(axes=[-1, -2])([H, W])
        H = Add(name='Dense_{}'.format(i))([H, B])
        # H = Activation(activation)(H)

        if i < len(indices) - 1:
            H = Activation(activation)(H)

    if output_activation != None:
        outp = Activation(output_activation)(H)
        return Model(inputs=[x, w], outputs=[outp], name="M")
    else:
        return Model(inputs=[x, w], outputs=[H], name="M")


class Hyper_Net_GAN(GAN):

    def __init__(self, main_net_config, g_input_dim, g_num_hidden, g_hidden_dim, c_num_hidden, c_hidden_dim,
                 sample_size=64, mode='gan'):
        self.main_net_config = main_net_config
        g_out_dim = weights_dim(main_net_config['units'])

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
        self.gan_model_tg.compile(optimizer=self.g_optimizer(),
                                  loss={'C_freezed': 'binary_crossentropy', 'M': neloglik_normal},
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

        # x = np.tile(x, (self.sample_size, 1)).reshape((self.sample_size, *x.shape))
        # y = np.tile(y, (self.sample_size, 1)).reshape((self.sample_size, *y.shape))

        self.train_critic(n_c_pretrain)

        for i in tqdm(range(n_train)):
            if i % c_train_interval == 0:
                self.train_critic(n_c_train_perinterval)
            self.train_critic(n_c_train_perit)
            # select = np.random.choice(x.shape[0],x.shape[0]//2)
            # print(select)
            # print(x[select,:])
            xx = x#[select,:]
            yy = y#[select,:]
            xx = np.tile(xx, (self.sample_size, 1)).reshape((self.sample_size, *xx.shape))
            yy = np.tile(yy, (self.sample_size, 1)).reshape((self.sample_size, *yy.shape))

            self.train_generator(xx, yy)

    def predict(self, x, z=None, sample_size=1):

        if z is None:
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
        else:
            sample_size = z.shape[0]

        X = np.tile(x, (sample_size, 1)).reshape((sample_size, *x.shape))

        return self.gan_model_pred.predict([z, X])


class Hyper_Net_WGAN(WGAN_gp):

    def __init__(self, main_net_config, g_input_dim, g_num_hidden, g_hidden_dim, c_num_hidden, c_hidden_dim,
                 sample_size=64, mode='gan'):
        self.main_net_config = main_net_config
        g_out_dim = weights_dim(main_net_config['units'])
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
        self.gan_model_tg.compile(optimizer=self.g_optimizer(),
                                  loss={'C_freezed': Wasserestein_loss, 'M': neloglik_normal},
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
            # select = np.random.choice(x.shape[0],x.shape[0]//2)
            xx = x#[select,:]
            yy = y#[select,:]
            xx = np.tile(xx, (self.sample_size, 1)).reshape((self.sample_size, *xx.shape))
            yy = np.tile(yy, (self.sample_size, 1)).reshape((self.sample_size, *yy.shape))
            self.train_generator(x, y)

    def predict(self, x, z=None, sample_size=1):

        if z is None:
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_input_dim,)), cov=np.eye(self.g_input_dim),
                                              size=self.sample_size)
        else:
            sample_size = z.shape[0]

        X = np.tile(x, (sample_size, 1)).reshape((sample_size, *x.shape))

        return self.gan_model_pred.predict([z, X])


class  EEGAN(GAN):

    def __init__(self,g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, e_num_hidden,
                 e_hidden_dim, sample_size=64,  mode='bayes_by_gaussian_dropout'):
        super(EEGAN, self).__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden,
                                    c_hidden_dim, sample_size,mode)

        self.e_num_hidden, self.e_hidden_dim = e_num_hidden, e_hidden_dim

    def encoder(self):
        x = Input(shape=(self.g_out_dim,))
        h= Hidden(ltype=LAYER_TYPES[self.mode], kernel_regularizer=(lambda: l1_l2(1e-5, 1e-5)), batch_norm=True)(inputs=x, dims=[self.c_hidden_dim] * self.c_num_hidden, name_prefix='E_h')
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
            reconstruction = 0.5* mean_squared_error(z,z_prime)
            return prior+reconstruction

        self.gan_model_tg.compile(optimizer=self.g_optimizer(), loss=eegan_loss)

class EEWGAN_gp(WGAN_gp):

    def __init__(self,g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden, c_hidden_dim, e_num_hidden,
                 e_hidden_dim, sample_size=64,  mode='bayes_by_gaussian_dropout'):
        super(EEWGAN_gp, self).__init__(g_input_dim, g_num_hidden, g_hidden_dim, g_out_dim, c_num_hidden,
                                        c_hidden_dim, sample_size,mode)

    def encoder(self):
        x = Input(shape=(self.g_out_dim,))
        h= Hidden(ltype=LAYER_TYPES[self.mode], kernel_regularizer=(lambda: l1_l2(1e-5, 1e-5)), batch_norm=True)(inputs=x, dims=[self.c_hidden_dim] * self.c_num_hidden, name_prefix='E_h')
        z = Dense(self.g_out_dim, name='e_out')(h)
        return Model(inputs=x,outputs=z)

    def build_gan_model(self):
        # model to train generator
        z = Input(shape=(self.g_input_dim,), name='g_in')
        x = self.g_model(z)
        encoder = self.encoder()

        z_prime = encoder(x)

        c_out = self.c_freezed(x)
        self.gan_model_tg = Model(inputs=[z], outputs=[c_out])

        def eewgan_loss(y_true,y_pred):
            prior = Wasserestein_loss(y_true,y_pred)
            reconstruction = mean_squared_error(z,z_prime)
            return prior+reconstruction
        #
        # self.gan_model_tg.compile(optimizer=self.g_optmizer(), loss=Wasserestein_loss)
        self.gan_model_tg.compile(optimizer=self.g_optimizer(), loss=eewgan_loss)
