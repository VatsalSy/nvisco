import jax.numpy as np
import numpy as onp
from NODE_fns import NODE as NODE
from jax import grad, random, jit, vmap
from functools import partial
from jax.experimental import optimizers
import pickle
import matplotlib.pyplot as plt
key = random.PRNGKey(0)

#----------------------Generate Training Data--------------------------#
def dphidtaui_gov(tau_i, etad = 1360, etav = 175000): 
    tau1, tau2, tau3 = tau_i
    trtau = tau1 + tau2 + tau3
    dphidtau1 = 2*(1/3/etad)*trtau - 1/etad*(tau2+tau3)
    dphidtau2 = 2*(1/3/etad)*trtau - 1/etad*(tau1+tau3)
    dphidtau3 = 2*(1/3/etad)*trtau - 1/etad*(tau1+tau2)
    return [dphidtau1, dphidtau2, dphidtau3]
taui = onp.mgrid[-200:200:10j, -200:200:10j, -200:200:10j]
taui = taui.reshape([3,-1]).transpose()

taui = -onp.sort(-taui)

dphidtaui = onp.zeros_like(taui)
for i in range(taui.shape[0]):
    dphidtaui[i] = dphidtaui_gov(taui[i])

tau1 = taui[:,0]
tau2 = taui[:,1]
tau3 = taui[:,2]
I = tau1**2 + tau2**2 + tau3**2 - tau1*tau2 - tau1*tau3 - tau2*tau3

inp_std = onp.std(I)
out_std = 1/3/1360

#--------------Initialize the parameters of the model------------------#
def init_params(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    b = np.zeros(layers[i + 1])
    return Ws, b
    
layers = [1, 5, 5, 1]
params = init_params(layers, key)


#-------------A function to predict dPhidtaui given taui---------------#
def dPhi(params, tau1, tau2, tau3):

    # N1 = NODE(tau1, NODE1_params)
    # N2 = NODE(tau1 + tau2, NODE2_params)
    # N3 = NODE(tau1 + tau2 + tau3, NODE3_params)
    # N4 = NODE(tau1**2 + tau2**2 + tau3**2 + 2*tau1*tau2 + 2*tau1*tau3 + 2*tau2*tau3, params) #I1^2
    I = tau1**2 + tau2**2 + tau3**2 - tau1*tau2 - tau1*tau3 - tau2*tau3
    I = I/inp_std
    N5 = NODE(I, params) #I1^2 - 3I2
    N5 = N5*out_std

    N1 = N2 = N3 = N4 = 0

    Phi1 = N1 + N2 + N3 + 2*N4*(tau1 + tau2 + tau3) + N5*(2*tau1 - tau2 - tau3) #dphi/dtau1
    Phi2 =      N2 + N3 + 2*N4*(tau1 + tau2 + tau3) + N5*(2*tau2 - tau1 - tau3)
    Phi3 =           N3 + 2*N4*(tau1 + tau2 + tau3) + N5*(2*tau3 - tau1 - tau2)

    return Phi1, Phi2, Phi3
dPhi_vmap = vmap(dPhi, in_axes=(None, 0, 0, 0), out_axes = (0, 0, 0))

#-----------------------------Training----------------------------------#
#@jit
def loss(params, taui, dphidtaui_gt):
    tau1, tau2, tau3 = taui.transpose()
    dphidtaui_pr = dPhi_vmap(params, tau1, tau2, tau3)
    loss = np.average((dphidtaui_pr[0]-dphidtaui_gt[:,0])**2)
    loss+= np.average((dphidtaui_pr[1]-dphidtaui_gt[:,1])**2)
    loss+= np.average((dphidtaui_pr[2]-dphidtaui_gt[:,2])**2)
    return loss


@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, X_batch, Y_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)


def train(loss, X, Y, opt_state, key, nIter = 5000, batch_size = 100):
    global best_params
    lowest_loss = 10.0
    train_loss = []
    for it in range(nIter+1):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch], Y[idx_batch])
        if it % 1000 == 0 or it == nIter:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss

opt_init, opt_update, get_params = optimizers.adam(1.e-4)
# # The following is temporary (for the loss defined in log space)
# opt_init, opt_update, get_params = optimizers.adam(5.e-5)
opt_state = opt_init(params)


params, train_loss = train(loss, taui, dphidtaui, opt_state, key, nIter=200000)
with open('saved/phi_params_NI12m3I2.npy', 'wb') as f:
    pickle.dump(params, f)
with open('saved/phi_norm_w_NI12m3I2.npy', 'wb') as f:
    pickle.dump([inp_std, out_std], f)

fig,ax = plt.subplots()
ax.plot(train_loss)
ax.set_yscale('log')
fig.savefig('figs/train_Phi_NI12m3I2.jpg')



#---------------------------Testing----------------------------------#
n = 1000
taui = onp.random.normal(size=[n,3])*200
taui = -onp.sort(-taui) #Sort taui in descending order
dphidtaui = onp.zeros_like(taui)
for i in range(n):
    dphidtaui[i] = dphidtaui_gov(taui[i])

l = loss(params, taui, dphidtaui)
print(l)

dphidtaui = dphidtaui.transpose()
dphidtaui_pr = dPhi_vmap(params, taui[:,0], taui[:,1], taui[:,2])
dphidtaui_pr = onp.array(dphidtaui_pr)

fig,ax = plt.subplots(1,3, figsize=[16,4])
names = [1,2,3]
for i in range(3):
    mx = onp.max(dphidtaui[i])
    mn = onp.min(dphidtaui[i])
    ax[i].plot([mn, mx], [mn, mx], 'r')
    ax[i].plot(dphidtaui[i], dphidtaui_pr[i], 'k.')
    ax[i].set(title='$\\partial \\Phi / \\partial \\tau_{{ {s} }}$'.format(s = names[i]))
    
ax[0].set(xlabel='Govindjee', ylabel='Prediction')
fig.savefig('figs/test_Phi_NI12m3I2.jpg')

print(onp.mean(dphidtaui), onp.mean(dphidtaui_pr))








