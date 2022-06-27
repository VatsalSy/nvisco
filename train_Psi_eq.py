#Train the Neural ODE based Psi_EQ with the last points of stress data
import jax.numpy as np
import numpy as onp
from NODE_fns import sigma_biaxial_vmap
from jax import grad, random, jit
from functools import partial
from jax.experimental import optimizers
import pickle
import matplotlib.pyplot as plt
key = random.PRNGKey(0)
from jax.config import config
config.update('jax_disable_jit', False)



def init_params(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws

layers = [1, 5, 5, 1]
I1_params = init_params(layers, key)
I2_params = init_params(layers, key)
Iv_params = init_params(layers, key)
Iw_params = init_params(layers, key)
J1_params = init_params(layers, key)
J2_params = init_params(layers, key)
J3_params = init_params(layers, key)
J4_params = init_params(layers, key)
J5_params = init_params(layers, key)
J6_params = init_params(layers, key)
I_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
theta = 1.0
Psi1_bias = -3.0
Psi2_bias = -3.0

params = (I1_params, I2_params, Iv_params, Iw_params, J1_params, J2_params, J3_params, J4_params, J5_params, J6_params, I_weights, \
          theta, Psi1_bias, Psi2_bias)

opt_init, opt_update, get_params = optimizers.adam(3.e-5)
opt_state = opt_init(params)

def loss(params, lmb, sigma_gt):
    lm1 = lmb[:,-1,0]
    lm2 = lmb[:,-1,1]
    sigma_pr = sigma_biaxial_vmap(lm1, lm2, params)  #Compare only the last point in the stress-stretch curve
    loss1 = np.average((sigma_pr[:,0,0]-sigma_gt[:,-1,0])**2) 
    loss2 = np.average((sigma_pr[:,1,1]-sigma_gt[:,-1,1])**2)
    loss = (loss1+loss2)/2
    return  loss

@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, X_batch, Y_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)

def train(loss, X, Y, opt_state, key, nIter = 10000, batch_size = 10):
    train_loss = []
    val_loss = []
    for it in range(nIter):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch], Y[idx_batch])         
        if (it+1)% 1000 == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it+1, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, val_loss


with open('training_data/gov_data.npy','rb') as f:
    time, lmb_x, lmb_y, sgm_x, sgm_y = onp.load(f)
sigma_gt = np.transpose(np.array([sgm_x, sgm_y]), axes=[1,2,0]) #[n_curve, n_pt, 2]
lmb = np.transpose(np.array([lmb_x, lmb_y]), axes=[1,2,0])
params, train_loss, val_loss = train(loss,lmb, sigma_gt, opt_state, key, nIter = 500000, batch_size = 10)

sigma_pr = sigma_biaxial_vmap(lmb_x[:,-1], lmb_y[:,-1], params)
xmin,xmax = np.min(sigma_gt[:,-1,0]), np.max(sigma_gt[:,-1,0])
ymin,ymax = np.min(sigma_gt[:,-1,1]), np.max(sigma_gt[:,-1,1])

fig,ax = plt.subplots(1,2, figsize=[10,4])
ax[0].plot([xmin, xmax], [xmin, xmax], 'r-')
ax[0].plot(sigma_gt[:,-1,0],sigma_pr[:, 0,0],'.')
ax[1].plot([ymin, ymax], [ymin, ymax], 'r-')
ax[1].plot(sigma_gt[:,-1,1],sigma_pr[:, 1,1],'.')

fig.suptitle(r"Results of training $\Psi_{EQ}^{NODE}$ with the last point of each stress-stretch curve")
ax[0].set(xlabel="Ground truth $\sigma_x$", ylabel='Predicted $\sigma_x$', aspect='equal', xlim=[xmin, xmax], ylim=[xmin, xmax])
ax[1].set(xlabel="Ground truth $\sigma_y$", ylabel='Predicted $\sigma_y$', aspect='equal', xlim=[ymin, ymax], ylim=[ymin, ymax])
fig.savefig('figs/train_Psi_eq.jpg')

with open('saved/Psi_eq_params.npy', 'wb') as f:
    pickle.dump(params, f)