import matplotlib.pyplot as plt
from models import *
import os
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)
modes= ['basic', 'deep_implicit', 'bayes_by_gaussian_dropout']
models = [GAN, WGAN_gp]
num_hiddens = [3,5,8]

# gan = WGAN_gp( g_input_dim = 2, g_num_hidden= 5, g_hidden_dim = 10, g_out_dim = 2 , c_num_hidden = 4, c_hidden_dim=30 ,sample_size= 64, mode='deep_implicit')

n_train = 1000

def run(directory, model_type , num_hidden,mode):
    if not os.path.exists(directory):
        os.makedirs(directory)
    gan = model_type( g_input_dim = 2, g_num_hidden= num_hidden, g_hidden_dim = 10, g_out_dim = 2 , c_num_hidden =num_hidden, c_hidden_dim=20 ,sample_size= 64, mode=mode)
    for iii in range(10):
        n_c_pretrain=50
        if iii == 0:
            n_c_pretrain=100
        gan.train(n_train=n_train, n_c_pretrain=n_c_pretrain, n_c_train_perit=5, n_c_train_perinterval =200, c_train_interval = 500 )
        f =gan.sample_fake(8*400)
        r =gan.sample_real(size=400)
        plt.figure()
        plt.ylim(-13,13)
        plt.xlim(-13,13)
        b=plt.scatter(r[:,0],r[:,1],alpha=0.3)
        a=plt.scatter(f[:,0],f[:,1],alpha=0.2)
        plt.xlabel('After {} epochs'.format((iii+1)*n_train))
        plt.legend((a,b),('fake', 'real'),scatterpoints=1,bbox_to_anchor=(1.35,.0),loc='lower right',fontsize=14)
        plt.savefig(directory+'/f{0}.png'.format(iii+1))


#
# for mode in modes:
#     for num_hidden in num_hiddens:
#         for model_type in models:
#             directory = './picture/'+ model_type.__name__+'/'+mode+'_'+str(num_hidden)
#             run(directory, model_type, num_hidden,mode)

# model_type = GAN
# mode = 'basic'
num_hidden = 3
# directory = './picture/'+ model_type.__name__+'/'+mode+'_'+str(num_hidden)
# run(directory, model_type, num_hidden,mode)


for mode in modes:
#     for num_hidden in num_hiddens:
    for model_type in models:
        directory = './picture/'+ model_type.__name__+'/'+mode+'_'+str(num_hidden)
        run(directory, model_type, num_hidden,mode)
