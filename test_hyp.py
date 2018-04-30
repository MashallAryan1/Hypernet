import matplotlib.pyplot as plt
from hypernets import *
import os
import numpy as np
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
from sklearn.preprocessing import  StandardScaler
seed(1)
# modes= ['basic', 'deep_implicit', 'bayes_by_gaussian_dropout']
modes= ['deep_implicit','bayes_by_gaussian_dropout', 'basic','deepst_n_gaussian']
models = [ Hyper_Net_GAN,Hyper_Net_WGAN,]
# modes= ['deep_implicit',]
# models = [ Hyper_Net_WGAN,]

#

size=100
sample_size = 64
n_train = 500
left,right = 0, 30

# z_dim = 15

# z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size )
# xx=np.tile(x,(size,1)).reshape((size,*x.shape))

xxx = np.linspace(left,right,500).reshape((500,1))
yyy= np.sin(xxx)

main_net_config ={}
main_net_config['units'] = [1,10,1]
main_net_config['activation'] = 'tanh'
main_net_config['output_activation'] = None
# input_dim=z_dim#weights_dim(main_net_config['units'])
# print(input_dim)


# gan = Hyper_Net_GAN( main_net_config, g_input_dim =input_dim, g_num_hidden= 8, g_hidden_dim =100, g_out_dim = 2 , c_num_hidden = 8, c_hidden_dim=20 ,sample_size= 100 ,mode = 'deep_implicit')

# gan.build_model()
# directory ='./picture/GAN_BNN_ST'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# z = np.random.multivariate_normal(mean=np.zeros(shape=(input_dim,)), cov=np.eye(input_dim), size=size )
def run(model_type, funct, mode, num_hidden, z_dim,x,y):
    g_out_dim = weights_dim(main_net_config['units'])
    g_h_dim= 10 * g_out_dim
    gan = model_type(main_net_config, g_input_dim= z_dim, g_num_hidden=num_hidden//2, g_hidden_dim=g_h_dim,
                        c_num_hidden=num_hidden, c_hidden_dim=g_out_dim*2//3, sample_size=sample_size, mode= mode)
    gan.sample_real = (lambda sample_size: np.random.multivariate_normal(mean=np.zeros(shape=(g_out_dim,)), cov=np.eye(g_out_dim), size=sample_size ))


    gan.build_model()
    directory = './picture1/'+str(funct)+"/"+model_type.__name__+'_'+mode+str(num_hidden)+"_"+str(z_dim)
    if not os.path.exists(directory):
        os.makedirs(directory)

    z = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=np.eye(z_dim), size=size)
    z1 = np.random.multivariate_normal(mean=np.zeros(shape=(z_dim,)), cov=2*np.eye(z_dim), size=size)


    scaler = StandardScaler()
    x_= scaler.fit_transform(x)
    for iii in range(10):
        n_c_pretrain=20
        if iii == 0:
            n_c_pretrain=100
        gan.train(x_, y, n_train=n_train,  n_c_pretrain= n_c_pretrain ,n_c_train_perit=20, n_c_train_perinterval = 50, c_train_interval = 300)
        xx = scaler.transform(xxx)
        t=gan.predict(xx, z)
        t1=gan.predict(xx, z1)
        fig=plt.figure()
    #     plt.ylim(-2,2)
        plt.xlim(left,right)
        for i in range(size):
            plt.plot(xxx,t[i,:],color='b', alpha=0.3)
            plt.plot(xxx, t1[i, :], color='red', alpha=0.1)
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
num_hiddens = [3,5,8]

z_dims = [5,10,]
# z_dims = [10,]

for funct in range(3):
    x = np.random.uniform(left+10,right-10, 6).reshape((6, 1))  # np.linspace(-1,10,6).reshape((6,1))
    y = np.sin(x)
    for num_hidden in num_hiddens:
        for z_dim in z_dims:
            for mode in modes:
                        for model_type in models:
                            run(model_type, funct, mode, num_hidden, z_dim,x,y)
                            print("==")

