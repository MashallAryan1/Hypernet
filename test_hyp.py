import matplotlib.pyplot as plt
from models import *
import os
import numpy as np
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)
# modes= ['basic', 'deep_implicit', 'bayes_by_gaussian_dropout']
modes= ['deep_implicit', 'bayes_by_gaussian_dropout']
models = [ Hyper_Net_WGAN,Hyper_Net_GAN,]

#
size=100
# z_dim = 15

# z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size )
# xx=np.tile(x,(size,1)).reshape((size,*x.shape))

xxx = np.linspace(-1,10,5).reshape((5,1))
yyy= np.sin(xxx)

main_net_config ={}
main_net_config['units'] = [1,30,1]
main_net_config['activation'] = 'tanh'
main_net_config['output_activation'] = 'linear'
# input_dim=z_dim#weights_dim(main_net_config['units'])
# print(input_dim)


# gan = Hyper_Net_GAN( main_net_config, g_input_dim =input_dim, g_num_hidden= 8, g_hidden_dim =100, g_out_dim = 2 , c_num_hidden = 8, c_hidden_dim=20 ,sample_size= 100 ,mode = 'deep_implicit')

# gan.build_model()
# directory ='./picture/GAN_BNN_ST'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# z = np.random.multivariate_normal(mean=np.zeros(shape=(input_dim,)), cov=np.eye(input_dim), size=size )
def run(model_type, funct, mode, num_hidden, z_dim):
    print(model_type.__name__)

    gan = model_type(main_net_config, g_input_dim= z_dim, g_num_hidden=num_hidden, g_hidden_dim=100, g_out_dim=2,
                        c_num_hidden=num_hidden, c_hidden_dim=20, sample_size=64, mode= mode)

    gan.build_model()
    directory = './picture1/'+str(funct)+"/"+model_type.__name__+'_'+mode+str(num_hidden)+"_"+str(z_dim)
    if not os.path.exists(directory):
        os.makedirs(directory)

    z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size)

    for iii in range(0):
        n_c_pretrain=20
        if iii == 0:
            n_c_pretrain=100
        gan.train(x, y, n_train=500,  n_c_pretrain= n_c_pretrain ,n_c_train_perit= 20, n_c_train_perinterval = 50, c_train_interval = 300 )
        t=gan.predict(xxx, z )
        plt.figure()
    #     plt.ylim(-2,2)
        plt.xlim(-1,10)
        for i in range(size):
            plt.plot(xxx,t[i,:],color='b', alpha=0.3)
        plt.plot(xxx,yyy, color='r')
        plt.plot(x,y, 'ro')
    #     plt.xlabel('GAN after {} epochs'.format((iii+1)*500))
    #     plt.legend((a,b),('fake', 'real'),scatterpoints=1,bbox_to_anchor=(1.35,.0),loc='lower right',fontsize=14)
        plt.savefig(directory+'/f_{0}.png'.format(iii+1))
#
num_hiddens = [3,5,8]
num_hiddens = [3,5,8]

z_dims = [5,10,]
num_hidden=5
z_dim =10
for funct in range(3):
    x = np.random.uniform(-1, 10, 6).reshape((6, 1))  # np.linspace(-1,10,6).reshape((6,1))
    y = np.sin(x)

    for mode in modes:
        # for num_hidden in num_hiddens:
        #         for z_dim in z_dims:
                    for model_type in models:

                        run(model_type, funct, mode, num_hidden, z_dim)

