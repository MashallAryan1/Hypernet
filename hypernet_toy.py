from keras.optimizers import Adam
import keras.utils.vis_utils as vis
from generative_models import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils import *
from hypernet import *


tfd = tf.contrib.distributions

def sample_real(dim):
    return tfd.MultivariateNormalDiag(loc=np.zeros((dim,)), scale_diag=np.ones((dim,)))



import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
size=64
sample_size = 64
n_train = 200
left,right = 0, 30

# z_dim = 15

# z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size )
# xx=np.tile(x,(size,1)).reshape((size,*x.shape))

xxx = np.linspace(left,right,500).reshape((500,1))
yyy= np.sin(xxx)



main_net_config ={}
main_net_config['dims'] = [1,10,1]
main_net_config['activation'] = 'tanh'
main_net_config['output_activation'] = None
main_net_config['loss'] = main_net_neg_loglikelihood

g_config = {}
g_config['h_type'] ='DENSE' #'DEEP_IMPLICIT'
g_config['activation'] = 'leakyrelu'
g_config['output_activation'] = None
g_config['kernel_regularizer'] = (lambda: tf.keras.regularizers.l1_l2(1e-4, 1e-4))
g_config['batch_norm'] = True
g_config['dropout_rate'] = 0.0
g_config['dims'] = [2] + [20]*5 + [2]
g_config['loss'] = generator_loss_logit
g_config['optimizer'] =  (lambda :Adam())
c_config = {}
c_config['h_type'] = 'DENSE'
c_config['activation'] = 'leakyrelu'
c_config['output_activation'] = 'linear'
c_config['kernel_regularizer'] = (lambda: tf.keras.regularizers.l1_l2(1e-4, 1e-4))
c_config['batch_norm'] = False
c_config['dropout_rate'] = 0.1
c_config['dims'] = [2] + [50]*10 + [1]
c_config['loss'] = (binary_crossentropy_from_logit,binary_crossentropy_from_logit)
c_config['loss'] = (hing_fake,hing_real)
c_config['loss'] = {"fake":hing_fake,"real":hing_real}

c_config['optimizer'] = (lambda :Adam())
g_test = Hypernet(g_config, c_config, main_net_config, noise_sample_size=64,real_samples=sample_real())


import keras.utils.vis_utils as vis

vis.plot_model(g_test.g_model,  to_file='g_model.png')
vis.plot_model(g_test.c_model,  to_file='c_model.png')
vis.plot_model(g_test.g_freezed,  to_file='g_freezed.png')
vis.plot_model(g_test.c_freezed,  to_file='c_freezed.png')
vis.plot_model(g_test.gan_model_tc,  to_file='gan_model_tc.png')
vis.plot_model(g_test.gan_model_tg,  to_file='gan_model_tg.png')




def run( funct, z_dim,x,y,points):
    g_out_dim = weights_dim(main_net_config['dims'])
    gan = Hypernet(g_config, c_config,main_net_config, noise_sample_size=64,real_samples=sample_real(g_out_dim))

    directory = "./picture1/"+ str(funct) + "/"+ str(z_dim)+"_"+str(points)+"_50_500"
    if not os.path.exists(directory):
        os.makedirs(directory)

    z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size)


    scaler = StandardScaler()
    x_= scaler.fit_transform(x)
    for iii in range(20):
        n_c_pretrain=20
        if iii == 0:
            n_c_pretrain=100
        # gan.train(x_, y, n_train=n_train,  n_c_pretrain=n_c_pretrain ,n_c_train_perit=5, n_c_train_perinterval=10, c_train_interval=500)
        gan.train(x_, y, n_train, n_c_train=10)

        xx = scaler.transform(xxx)

        t = gan.predict(xx,z)
        # t1 = gan.predict(xx, z1)
        fig=plt.figure()
    #     plt.ylim(-2,2)
        plt.xlim(left,right)
        for i in range(size-30):
        #     # plt.plot(xxx, t1[i, :], color='red', alpha=0.1)
            plt.plot(xxx,t[i,:],color='b', alpha=0.2)
        plt.plot(xxx,yyy, color='r')
        plt.plot(x,y, 'ro')
    #     plt.xlabel('GAN after {} epochs'.format((iii+1)*500))
    #     plt.legend((a,b),('fake', 'real'),scatterpoints=1,bbox_to_anchor=(1.35,.0),loc='lower right',fontsize=14)
        plt.savefig(directory+'/f_{0}.png'.format(iii+1))
    #     plt.show(block=True)
        fig.clf()
        plt.close()


#
# num_hiddens = [8,]
num_hiddens = [5]#[3,5,8]

z_dims = [2]#[2,5,10,]
# z_dims = [10,]
n_points = 20
for points in range(5,n_points,2):
    for funct in range(10):
        x = np.random.uniform(left+10,right-10, points).reshape((points, 1))  # np.linspace(-1,10,6).reshape((6,1))
        y = np.sin(x)
        for num_hidden in num_hiddens:
            for z_dim in z_dims:
                        run(funct, z_dim,x,y,points)
                        print("==")

