from utils import *
from keras.layers import Input, Activation, Reshape, Dot, Add, Lambda
from keras.models import Model
from keras.optimizers import Adam
from generative_models import GAN
from tqdm import tqdm




class Hypernet(GAN):

    def __init__(self,g_config,c_config, m_config,noise_sample_size=64, real_samples=None, fake_label=0, real_label=1):
        self.m_config = m_config

        main_net_size=weights_dim(self.m_config['dims'])
        g_config['dims'][-1]= c_config['dims'][0]=main_net_size
        super().__init__(g_config, c_config, noise_sample_size, real_samples, fake_label, real_label)

    @staticmethod
    def build_main_net(config=None):
        dims, activation, output_activation = config['dims'], config['activation'], config['output_activation']

        indices = get_weight_indices(dims)
        start_w, stop_w, start_b, stop_b = indices[0]
        x = Input(shape=(None, dims[0],), name="X")
        w = Input(shape=(weights_dim(dims),), name="W_raw")

        W = Lambda(slice_weights, arguments={'start': start_w, 'stop': stop_w}, name='MainW_0_sliced')(w)
        W = Reshape((dims[0], dims[1]), name='MainW_0')(W)
        B = Lambda(slice_weights, arguments={'start': start_b, 'stop': stop_b}, name='MainB_0')(w)

        H = Dot(axes=[-1, -2])([x, W])
        H = Add(name='Dense_0')([H, B])
        H = Activation(activation)(H)

        for i in range(1, len(indices)):
            start_w, stop_w, start_b, stop_b = indices[i]
            W = Lambda(slice_weights, arguments={'start': start_w, 'stop': stop_w}, name='MainW_{}_sliced'.format(i))(w)
            W = Reshape((dims[i], dims[i + 1]), name='MainW_{}'.format(i))(W)
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


    def build_gan_model(self,z_tensor):
        """
         wire up generator and non-trainable discriminator/critic to  train generator
        :return: trainable GAN model self.gan_model_tg
        """
        ### model to train generator###

        z =  Input(shape=(self.g_config["dims"][0],),tensor=z_tensor)
        g_of_z = self.g_model(z)
        c_out = self.c_freezed(g_of_z)
        # self.gan_model_tg = Model(inputs=z,outputs=c_out)

        x = Input(shape=(None, self.m_config['dims'][0],),name='net_in')
        main_net = self.build_main_net(self.m_config)
        main_net_out = main_net([x,g_of_z])
        self.gan_model_tg = Model(inputs=[z,x],outputs=[c_out,main_net_out])
        self.gan_model_pred = Model(inputs=self.g_freezed.inputs+[x],outputs= main_net([x]+self.g_freezed.outputs))
        # g_of_z_freezed = self.g_freezed(z)
        # self.gan_model_freezed = Model(inputs=z,outputs=g_of_z_freezed)


    def compile_g(self):
        self.gan_model_tg.compile(optimizer=self.g_config['optimizer'](), loss={'C_freezed':self.g_config['loss'],'M':self.m_config['loss']},loss_weights={'C_freezed':0.5,'M':0.5})

    def evaluate(self,x,y,steps=1):
        real_labels = self.get_real_labels(self.noise_sample_size)
        x = np.tile(x, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *x.shape))
        y = np.tile(y, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *y.shape))
        self.performance_log['generator'].append(self.gan_model_tg.evaluate(x={'g_in':None, 'net_in': x},y=[real_labels,y],steps=1))

    def train(self,x,y, n_train, n_c_train, n_pretrain=0,noise_std=10e-5, early_stopping_threshold = -1):
        fake_labels = self.get_fake_labels(self.noise_sample_size)
        real_labels = self.get_real_labels(self.noise_sample_size)
        x = np.tile(x, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *x.shape))
        y = np.tile(y, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *y.shape))
        best_total_loss = np.Infinity
        early_stopping_counter = 0

        for j in range(n_pretrain):
            self.performance_log['critic'].append(
                self.gan_model_tc.train_on_batch(x=None, y=[fake_labels, real_labels]))

        for i in tqdm(range(n_train)):
            for j in range(n_c_train):
                self.performance_log['critic'].append(self.gan_model_tc.train_on_batch(x=None,y=[fake_labels, real_labels]))
            y_ = y
            if noise_std > 0:
                y_ = y+np.random.normal(0,noise_std,size=y.shape)
            self.performance_log['generator'].append(self.gan_model_tg.train_on_batch(x={'g_in':None, 'net_in': x},y=[real_labels,y_]))
            early_stopping_counter+=1
            # if self.performance_log['generator'][-1][2] < best_total_loss:
            #     best_total_loss = self.performance_log['generator'][-1][2]
            #     early_stopping_counter=0
            # if early_stopping_threshold > 0 and early_stopping_counter >= early_stopping_threshold and np.mean(self.performance_log['generator'][-10:-1][1]) >= self.performance_log['generator'][-1][1]:
            #     print(early_stopping_counter)
            #     break;

    def predict(self, x, z=None, sample_size=1):
        x = np.tile(x, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *x.shape))

        if z is None:
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_config["dims"][0],)), cov=np.eye(self.g_config["dims"][0]),
                                              size=self.noise_sample_size)

        return self.gan_model_pred.predict([z, x])
