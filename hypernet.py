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


    def compile_g(self):
        self.gan_model_tg.compile(optimizer=self.g_config['optimizer'](), loss={'C_freezed':self.g_config['loss'],'M':self.m_config['loss']})


    def train(self,x,y, n_train, n_c_train,n_pretrain=0):
        fake_labels = self.get_fake_labels(self.noise_sample_size)
        real_labels = self.get_real_labels(self.noise_sample_size)
        x = np.tile(x, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *x.shape))
        y= np.tile(y, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *y.shape))

        for j in range(n_pretrain):
            self.performance_log['critic'].append(
                self.gan_model_tc.train_on_batch(x=None, y=[fake_labels, real_labels]))

        for i in tqdm(range(n_train)):
            for j in range(n_c_train):
                self.performance_log['critic'].append(self.gan_model_tc.train_on_batch(x=None,y=[fake_labels, real_labels]))
            self.performance_log['generator'].append(self.gan_model_tg.train_on_batch(x={'g_in':None, 'net_in': x},y=[real_labels,y]))


    def predict(self, x, z=None, sample_size=1):
        x = np.tile(x, (self.noise_sample_size, 1)).reshape((self.noise_sample_size, *x.shape))

        if z is None:
            z = np.random.multivariate_normal(mean=np.zeros(shape=(self.g_config["dims"][0],)), cov=np.eye(self.g_config["dims"][0]),
                                              size=self.noise_sample_size)

        return self.gan_model_pred.predict([z, x])

#
#
# tfd = tf.contrib.distributions
#
# def sample_real(dim):
#     return tfd.MultivariateNormalDiag(loc=np.zeros((dim,)), scale_diag=np.ones((dim,)))
#     # a = np.linspace(0.0, 2 * np.pi, 9)
#     # a = a[:-1]
#     # r = 10
#     # centers_x = 1.0 + r * np.cos(a)
#     # centers_y = 1.0 + r * np.sin(a)
#     # means = list(zip(centers_x, centers_y))
#     # probs= [1.0/8.0]*8
#     # components = [tfd.MultivariateNormalDiag(loc=[centers_x, centers_y], scale_diag=[0.8, 0.8]) for centers_x, centers_y in means]
#     # # Create a mixture of two Gaussians:
#     # mix = 0.3
#     # bimix_gauss = tfd.Mixture(
#     #     cat=tfd.Categorical(probs=probs),
#     #     components=components)
#     # return bimix_gauss
#
#
#
#
#
# import matplotlib.pyplot as plt
# import os
# from sklearn.preprocessing import StandardScaler
# size=64
# sample_size = 64
# n_train = 200
# left,right = 0, 30
#
# # z_dim = 15
#
# # z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size )
# # xx=np.tile(x,(size,1)).reshape((size,*x.shape))
#
# xxx = np.linspace(left,right,500).reshape((500,1))
# yyy= np.sin(xxx)
#
#
#
# main_net_config ={}
# main_net_config['dims'] = [1,10,1]
# main_net_config['activation'] = 'tanh'
# main_net_config['output_activation'] = None
# main_net_config['loss'] = main_net_neg_loglikelihood
#
# g_config = {}
# g_config['h_type'] ='DENSE' #'DEEP_IMPLICIT'
# g_config['activation'] = 'leakyrelu'
# g_config['output_activation'] = None
# g_config['kernel_regularizer'] = (lambda: tf.keras.regularizers.l1_l2(1e-4, 1e-4))
# g_config['batch_norm'] = True
# g_config['dropout_rate'] = 0.0
# g_config['dims'] = [2] + [20]*5 + [2]
# g_config['loss'] = generator_loss_logit
# g_config['optimizer'] =  (lambda :Adam())
# c_config = {}
# c_config['h_type'] = 'DENSE'
# c_config['activation'] = 'leakyrelu'
# c_config['output_activation'] = 'linear'
# c_config['kernel_regularizer'] = (lambda: tf.keras.regularizers.l1_l2(1e-4, 1e-4))
# c_config['batch_norm'] = False
# c_config['dropout_rate'] = 0.1
# c_config['dims'] = [2] + [50]*10 + [1]
# c_config['loss'] = (binary_crossentropy_from_logit,binary_crossentropy_from_logit)
# c_config['loss'] = (hing_fake,hing_real)
# c_config['loss'] = {"fake":hing_fake,"real":hing_real}
#
# c_config['optimizer'] = (lambda :Adam())
# g_test = Hypernet(g_config, c_config, main_net_config, noise_sample_size=64,real_samples=sample_real())


import keras.utils.vis_utils as vis

# vis.plot_model(g_test.g_model,  to_file='g_model.png')
# vis.plot_model(g_test.c_model,  to_file='c_model.png')
# vis.plot_model(g_test.g_freezed,  to_file='g_freezed.png')
# vis.plot_model(g_test.c_freezed,  to_file='c_freezed.png')
# vis.plot_model(g_test.gan_model_tc,  to_file='gan_model_tc.png')
# vis.plot_model(g_test.gan_model_tg,  to_file='gan_model_tg.png')



#
# def run( funct, z_dim,x,y,points):
#     g_out_dim = weights_dim(main_net_config['dims'])
#     gan = Hypernet(g_config, c_config,main_net_config, noise_sample_size=64,real_samples=sample_real(g_out_dim))
#
#     directory = "./picture1/"+ str(funct) + "/"+ str(z_dim)+"_"+str(points)+"_50_500"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size)
#
#
#     scaler = StandardScaler()
#     x_= scaler.fit_transform(x)
#     for iii in range(20):
#         n_c_pretrain=20
#         if iii == 0:
#             n_c_pretrain=100
#         # gan.train(x_, y, n_train=n_train,  n_c_pretrain=n_c_pretrain ,n_c_train_perit=5, n_c_train_perinterval=10, c_train_interval=500)
#         gan.train(x_, y, n_train, n_c_train=10)
#
#         xx = scaler.transform(xxx)
#
#         t = gan.predict(xx,z)
#         # t1 = gan.predict(xx, z1)
#         fig=plt.figure()
#     #     plt.ylim(-2,2)
#         plt.xlim(left,right)
#         for i in range(size-30):
#         #     # plt.plot(xxx, t1[i, :], color='red', alpha=0.1)
#             plt.plot(xxx,t[i,:],color='b', alpha=0.2)
#         plt.plot(xxx,yyy, color='r')
#         plt.plot(x,y, 'ro')
#     #     plt.xlabel('GAN after {} epochs'.format((iii+1)*500))
#     #     plt.legend((a,b),('fake', 'real'),scatterpoints=1,bbox_to_anchor=(1.35,.0),loc='lower right',fontsize=14)
#         plt.savefig(directory+'/f_{0}.png'.format(iii+1))
#     #     plt.show(block=True)
#         fig.clf()
#         plt.close()
#
#
# #
# # num_hiddens = [8,]
# num_hiddens = [5]#[3,5,8]
#
# z_dims = [2]#[2,5,10,]
# # z_dims = [10,]
# n_points = 20
# for points in range(5,n_points,2):
#     for funct in range(10):
#         x = np.random.uniform(left+10,right-10, points).reshape((points, 1))  # np.linspace(-1,10,6).reshape((6,1))
#         y = np.sin(x)
#         for num_hidden in num_hiddens:
#             for z_dim in z_dims:
#                         run(funct, z_dim,x,y,points)
#                         print("==")
#
