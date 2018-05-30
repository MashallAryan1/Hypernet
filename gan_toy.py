from keras.optimizers import Adam
import keras.utils.vis_utils as vis
from generative_models import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from utils import *
tfd = tf.contrib.distributions

def sample_real():
    a = np.linspace(0.0, 2 * np.pi, 9)
    a = a[:-1]
    r = 10
    centers_x = 1.0 + r * np.cos(a)
    centers_y = 1.0 + r * np.sin(a)
    means = list(zip(centers_x, centers_y))
    probs= [1.0/8.0]*8
    components = [tfd.MultivariateNormalDiag(loc=[centers_x, centers_y], scale_diag=[0.8, 0.8]) for centers_x, centers_y in means]
    # Create a mixture of two Gaussians:
    mix = 0.3
    bimix_gauss = tfd.Mixture(
        cat=tfd.Categorical(probs=probs),
        components=components)
    return bimix_gauss


directory = './picture'

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
# c_config['loss'] = (binary_crossentropy_from_logit,binary_crossentropy_from_logit)
# c_config['loss'] = (hing_fake,hing_real)
c_config['loss'] = {"fake":hing_fake,"real":hing_real}

c_config['optimizer'] = (lambda :Adam())
g_test = GAN(g_config, c_config, noise_sample_size=64,real_samples=sample_real())






vis.plot_model(g_test.g_model,  to_file='g_model.png')
vis.plot_model(g_test.c_model,  to_file='c_model.png')
vis.plot_model(g_test.g_freezed,  to_file='g_freezed.png')
vis.plot_model(g_test.c_freezed,  to_file='c_freezed.png')
vis.plot_model(g_test.gan_model_tc,  to_file='gan_model_tc.png')
vis.plot_model(g_test.gan_model_tg,  to_file='gan_model_tg.png')






cm = LinearSegmentedColormap.from_list(name='mycm',colors=['darkgray','white'], N = 10000)
cm1 = LinearSegmentedColormap.from_list(name='mycm',colors=['orangered','lime'], N = 10000)
points = np.linspace(-15,15,300)
xv,yv= np.meshgrid(points,points)
test_points =np.vstack((xv.ravel(),yv.ravel())).T

n_train = 200
#
g_test.train(n_train=0, n_c_train=0, n_pretrain=100)

ss =sample_real().sample(400)

for i in range(50):

    g_test.train(n_train=n_train, n_c_train=2 )
    f = g_test.sample_fake(8*100)

    r = K.eval(ss)
    fake_loss = np.squeeze(g_test.get_loss(test_points))

    fig = plt.figure()
    plt.ylim(-15, 15)
    plt.xlim(-15, 15)
    # plt.imshow(xv,yv,fake_loss.reshape(300,300),vmin=0,vmax=1,interpolation='bilinear',extent=[-20, 20, -20, 20],cmap=cm1)
    plt.imshow(fake_loss.reshape(300, 300), extent=[-15, 15, -15, 15], cmap='gray', origin='lower')
    plt.contour(xv, yv, fake_loss.reshape(300, 300), extent=[-15, 15, -15, 15], cmap='autumn')
    #
    b = plt.scatter(r[:, 0], r[:, 1], alpha=0.4)
    a = plt.scatter(f[:, 0], f[:, 1], c=np.squeeze(g_test.get_loss(f)), vmin=fake_loss.min(), vmax=fake_loss.max(),
                    cmap='autumn', alpha=0.2)

    # # a=plt.scatter(f[:,0],f[:,1],alpha=0.2)

    plt.xlabel('After {} epochs'.format((i + 1) * n_train))
    plt.colorbar()
    plt.savefig(directory + '/f{0}.png'.format(i + 1))
    fig.clf()
    plt.close()
