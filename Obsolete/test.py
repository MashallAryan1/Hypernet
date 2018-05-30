import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Obsolete.hypernets import *
import os
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)

cm = LinearSegmentedColormap.from_list(name='mycm',colors=['darkgray','white'], N = 10000)
cm1 = LinearSegmentedColormap.from_list(name='mycm',colors=['orangered','lime'], N = 10000)
points = np.linspace(-15,15,300)
xv,yv= np.meshgrid(points,points)
test_points =np.vstack((xv.ravel(),yv.ravel())).T


def _sample_real(size):
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


# modes= ['dense', 'deep_implicit', 'bayes_by_gaussian_dropout','deepst_n_gaussian']
modes= ['dense',]



# models = [GAN, WGAN_gp]
models = [GAN]

# models = [EEGAN, EEWGAN_gp]




num_hiddens = [3,5,8]

# gan = WGAN_gp( g_input_dim = 2, g_num_hidden= 5, g_hidden_dim = 10, g_out_dim = 2 , c_num_hidden = 4, c_hidden_dim=30 ,sample_size= 64, mode='deep_implicit')

n_train = 200





def run(directory, model_type , num_hidden,mode):
    if not os.path.exists(directory):
        os.makedirs(directory)
    gan = model_type( g_input_dim = 2, g_num_hidden= num_hidden, g_hidden_dim = 20, g_out_dim = 2 , c_num_hidden =num_hidden, c_hidden_dim=100 ,sample_size= 64, mode=mode)
    gan.sample_real = _sample_real
    # gan = model_type( g_input_dim = 2, g_num_hidden= num_hidden, g_hidden_dim = 20, g_out_dim = 2 , c_num_hidden =num_hidden, c_hidden_dim=20,e_num_hidden= num_hidden//2, e_hidden_dim = 10 ,sample_size= 64, mode=mode)

    for iii in range(100):
        n_c_pretrain=10
        if iii == 0:
            n_c_pretrain=100
        gan.train(n_train=n_train, n_c_pretrain=n_c_pretrain, n_c_train_perit=15, n_c_train_perinterval =50, c_train_interval = 500 )

        f =gan.sample_fake(8*400)
        r =gan.sample_real(400)
        fake_loss =np.squeeze( gan.get_loss(test_points))



        fig =plt.figure()
        plt.ylim(-15,15)
        plt.xlim(-15,15)
        # plt.imshow(xv,yv,fake_loss.reshape(300,300),vmin=0,vmax=1,interpolation='bilinear',extent=[-20, 20, -20, 20],cmap=cm1)
        plt.imshow(fake_loss.reshape(300,300),extent=[-15,15, -15,15],cmap='gray',origin='lower')
        plt.contour(xv,yv,fake_loss.reshape(300,300),extent=[-15,15, -15,15],cmap='autumn')
        #
        b=plt.scatter(r[:,0],r[:,1],alpha=0.4)
        a=plt.scatter(f[:,0],f[:,1],c=np.squeeze(gan.get_loss(f)),vmin=fake_loss.min(),vmax=fake_loss.max(),cmap='autumn',alpha=0.2)

        # # a=plt.scatter(f[:,0],f[:,1],alpha=0.2)

        plt.xlabel('After {} epochs'.format((iii+1)*n_train))
        plt.colorbar()
        # plt.legend((a,b),('fake', 'real'),scatterpoints=1,bbox_to_anchor=(1.45,.0),loc='lower right',fontsize=14)
        plt.savefig(directory+'/f{0}.png'.format(iii+1))
        fig.clf()
        plt.close()


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


for num_hidden in num_hiddens:
    for mode in modes:
        for model_type in models:
            directory = './picture/'+ model_type.__name__+'/'+mode+'_'+str(num_hidden)
            run(directory, model_type, num_hidden,mode)
