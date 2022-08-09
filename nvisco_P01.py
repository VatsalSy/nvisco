# A standalone python file to 
# 1. Train Govindjee's viscoelasticity model on experimental data for Pig 01 (P01)
# 2. Pretrain $\Psi_{EQ}^{NODE}$ with the $\Psi_{EQ}$ from Govindjee
# 3. Pretrain $\Psi_{NEQ}^{NODE}$ with the $\Psi_{NEQ}$ from Govindjee
# 4. Pretrain $\Phi^{NODE}$ with the $\Phi$ from Govindjee 
# 5. Retrain the whole model with P01 stress-stretch data

# all in a single file for the P01 data case. It needs training_data/gov_data.npy and NODE_fns.py to function. 
# This file and nvisco_P01.ipynb contain essentially the same stuff, except for figures. 
# nvisco_P01.py was made to run this notebook on the cluster and instead of showing the figures it just saves them.

# Import and initialize
from jax.config import config
config.update('jax_platform_name', 'cpu') #Otherwise jax displays an annoying warning every time
config.update('jax_disable_jit', False)
config.update('jax_enable_x64', True)
#config.update('jax_debug_nans', True) #this checks outputs for nans everytime and reruns the function with non-optimized mode.


import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)
from NODE_fns import NODE, NODE_nobias, sigma_split as sigma_NODE, sigma_split_vmap
from jax import grad, random, jit, vmap
from functools import partial
import jax.example_libraries.optimizers as optimizers
from jax.lax import cond, scan
from jax.experimental.ode import odeint
from diffrax import diffeqsolve, ODETerm, SaveAt, Heun as mysolver
import pickle
key = random.PRNGKey(0)
import jax

# Various useful functions
# 2 different functions to initialize 2 different NN architectures
def init_params_positivebias(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    b = np.zeros(layers[i + 1])
    return Ws, b
def init_params_nobias(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws
def dPhi_gov(tau_i, etad = 1360, etav = 175000): 
    tau1, tau2, tau3 = tau_i
    trtau = tau1 + tau2 + tau3
    dphidtau1 = 2*(1/3/etad + 1/9/etav)*trtau - 1/etad*(tau2+tau3)
    dphidtau2 = 2*(1/3/etad + 1/9/etav)*trtau - 1/etad*(tau1+tau3)
    dphidtau3 = 2*(1/3/etad + 1/9/etav)*trtau - 1/etad*(tau1+tau2)
    return [dphidtau1, dphidtau2, dphidtau3]

# Neural ODE based dPhi/dtaui
def dPhi_NODE(taui, Phi_params):
    NODE1_params, NODE2_params, NODE3_params, NODE4_params, NODE5_params = Phi_params
    tau1 = taui[0]
    tau2 = taui[1]
    tau3 = taui[2]

    I1 = tau1
    I2 = tau1 + tau2
    I3 = tau1 + tau2 + tau3
    I4 = tau1**2 + tau2**2 + tau3**2 + 2*tau1*tau2 + 2*tau1*tau3 + 2*tau2*tau3
    I5 = tau1**2 + tau2**2 + tau3**2 -   tau1*tau2 -   tau1*tau3 -   tau2*tau3

    I1 = I1/inp_std1
    I2 = I2/inp_std2
    I3 = I3/inp_std3
    I4 = I4/inp_std4
    I5 = I5/inp_std5

    N1 = NODE_nobias(I1, NODE1_params)
    N2 = NODE_nobias(I2, NODE2_params)
    N3 = NODE(I3, NODE3_params)
    N4 = NODE(I4, NODE4_params) #I1^2
    N5 = NODE(I5, NODE5_params) #I1^2 - 3I2

    N1 = np.max(np.array([N1, 0]))
    N2 = np.max(np.array([N2, 0]))

    N1 = N1*out_std1
    N2 = N2*out_std2
    N3 = N3*out_std3
    N4 = N4*out_std4
    N5 = N5*out_std5

    Phi1 = N1 + N2 + N3 + 2*N4*(tau1 + tau2 + tau3) + N5*(2*tau1 - tau2 - tau3) #dphi/dtau1
    Phi2 =      N2 + N3 + 2*N4*(tau1 + tau2 + tau3) + N5*(2*tau2 - tau1 - tau3)
    Phi3 =           N3 + 2*N4*(tau1 + tau2 + tau3) + N5*(2*tau3 - tau1 - tau2)

    return [Phi1, Phi2, Phi3]
dPhi_vmap = vmap(dPhi_NODE, in_axes=(0, None), out_axes = (0))

def tau_NEQ_gov(lm1e, lm2e, lm3e, params):
    alpha_m, mu_m, K_m, eta_D, eta_V, K, mu = params
    Je = lm1e*lm2e*lm3e
    
    b1 = Je**(-2/3)*lm1e**2
    b2 = Je**(-2/3)*lm2e**2
    b3 = Je**(-2/3)*lm3e**2

    devtau1 = 0
    devtau2 = 0
    devtau3 = 0
    for r in range(3):
        e = alpha_m[r]/2
        devtau1 = devtau1 + mu_m[r]*(2/3*b1**e - 1/3*(b2**e + b3**e)) #(B8)
        devtau2 = devtau2 + mu_m[r]*(2/3*b2**e - 1/3*(b1**e + b3**e))
        devtau3 = devtau3 + mu_m[r]*(2/3*b3**e - 1/3*(b1**e + b2**e))
    devtau = np.array([devtau1, devtau2, devtau3])

    tau_NEQI = 3*(K_m*10000)/2*(Je**2-1) #(B8)
    tau_A = devtau + 1/3*tau_NEQI #(B8)
    return tau_A

def tau_NEQ_NODE(lm1, lm2, lm3, Psi_neq_params):
    J = lm1*lm2*lm3
    sigma_NEQ = sigma_NODE(lm1, lm2, lm3, Psi_neq_params)
    tau_NEQ = J*sigma_NEQ
    return tau_NEQ

def sigma_NEQ_gov(lm1e, lm2e, lm3e, params):
    tau_A = tau_NEQ_gov(lm1e, lm2e, lm3e, params)
    Je = lm1e*lm2e*lm3e
    # tau_A = tau_NEQ(lm1e, lm2e, lm3e, params)
    tau_NEQ = np.array([[tau_A[0], 0, 0],
                        [0, tau_A[1], 0],
                        [0, 0, tau_A[2]]]) #Since stress and strain are coaxial in the isotropic case

    sigma_NEQ = 1/Je*tau_NEQ
    return sigma_NEQ

def sigma_EQ_gov(lm1, lm2, lm3, params):
    alpha_m, mu_m, K_m, eta_D, eta_V, K, mu = params
    J = lm1*lm2*lm3
    b = np.array([[lm1**2, 0, 0],
                [0, lm2**2, 0],
                [0, 0, lm3**2]])
    sigma_EQ = mu/J*(b-np.eye(3)) + 2*(K*10000)*(J-1)*np.eye(3)
    return sigma_EQ

def sigma(inputs, params, useNODE):
    lm1, lm2, lm3, lm1e, lm2e, lm3e = inputs
    if useNODE: # use NODE
        Psi_eq_params, Psi_neq_params, Phi_params = params
        sigma_EQ  = sigma_NODE(lm1, lm2, lm3, Psi_eq_params)
        sigma_NEQ = sigma_NODE(lm1e, lm2e, lm3e, Psi_neq_params)
    else: # use Govindjee
        sigma_NEQ = sigma_NEQ_gov(lm1e, lm2e, lm3e, params)
        sigma_EQ = sigma_EQ_gov(lm1, lm2, lm3, params)
        
    sigma = sigma_NEQ + sigma_EQ
    return sigma
getsigma = vmap(sigma, in_axes=(0, None, None), out_axes=0)
dsigma33 = grad(lambda inputs, params, useNODE: sigma(inputs,params,useNODE)[2,2])

def yprime_biaxial(y, t, lm1dot, lm2dot, tpeak, params, useNODE):
    lm1, lm2, lm3, lm1e, lm2e, lm3e = y

    true_fun  = lambda t: np.array([lm1dot, lm2dot])
    false_fun = lambda t: np.array([   0.0,    0.0], dtype='float64')
    lm1dot, lm2dot = cond(t<tpeak, true_fun, false_fun, t)

    if useNODE: # use NODE 
        Psi_eq_params, Psi_neq_params, Phi_params = params
        tau_A = tau_NEQ_NODE(lm1e, lm2e, lm3e, Psi_neq_params) 
        tau_A = np.array([tau_A[0,0], tau_A[1,1], tau_A[2,2]])
        dphidtaui = dPhi_NODE(tau_A, Phi_params)
    else: # use Govindjee
        alpha_m, mu_m, K_m, eta_D, eta_V, K, mu = params
        tau_A = tau_NEQ_gov(lm1e, lm2e, lm3e, params) 
        dphidtaui = dPhi_gov(tau_A, eta_D, eta_V)

    lm1edot = (lm1dot/lm1 - 0.5*dphidtaui[0])*lm1e
    lm2edot = (lm2dot/lm2 - 0.5*dphidtaui[1])*lm2e

    d = dsigma33([lm1,lm2,lm3,lm1e,lm2e,lm3e], params, useNODE)
    A = -(d[0]*lm1dot + d[1]*lm2dot + d[3]*lm1edot + d[4]*lm2edot)/d[2]
    B = -d[5]/d[2]

    Apr = A/lm3
    Bpr = B/lm3

    lm3edot = (Apr - 0.5*dphidtaui[2])/(1-Bpr*lm3e)*lm3e
    lm3dot = A + B*lm3edot
    return lm1dot, lm2dot, lm3dot, lm1edot, lm2edot, lm3edot

def yprime_triaxial(y, t, lm1dot, tpeak, params, useNODE):
    lm1, lm2, lm3, lm1e, lm2e, lm3e = y

    true_fun  = lambda t: np.array(lm1dot)
    false_fun = lambda t: np.array(   0.0)
    lm1dot = cond(t<tpeak, true_fun, false_fun, t)

    if useNODE: # use NODE 
        Psi_eq_params, Psi_neq_params, Phi_params = params
        tau_A = tau_NEQ_NODE(lm1e, lm2e, lm3e, Psi_neq_params) 
        tau_A = np.array([tau_A[0,0], tau_A[1,1], tau_A[2,2]])
        dphidtaui = dPhi_NODE(tau_A, Phi_params)
    else: # use Govindjee
        alpha_m, mu_m, K_m, eta_D, eta_V, K, mu = params
        tau_A = tau_NEQ_gov(lm1e, lm2e, lm3e, params) 
        dphidtaui = dPhi_gov(tau_A, eta_D, eta_V)

    lm1edot = (lm1dot/lm1 - 0.5*dphidtaui[0])*lm1e
    lm2edot = np.array(-0.5*dphidtaui[1]*lm2e)
    lm3edot = np.array(-0.5*dphidtaui[2]*lm3e)

    lm2dot = lm3dot = np.array(0)

    return lm1dot, lm2dot, lm3dot, lm1edot, lm2edot, lm3edot

@partial(jit, static_argnums=(1,))
def biaxial_visco(params, useNODE, time, lm1, lm2):
    ipeak1 = np.argmax(np.abs(np.around(lm1, 3)-1.0)) #around(lm1, 3) evenly rounds lm1 to 3 decimals
    ipeak2 = np.argmax(np.abs(np.around(lm2, 3)-1.0))
    ipeak = np.max(np.array([ipeak1,ipeak2]))
    tpeak = time[ipeak]
    lm1peak = lm1[ipeak]
    lm2peak = lm2[ipeak]

    lm1dot = (lm1peak-1.0)/tpeak
    lm2dot = (lm2peak-1.0)/tpeak
    
    yprime = lambda t, y, args: np.array(yprime_biaxial(y,t,lm1dot,lm2dot,tpeak,params,useNODE))
    term = ODETerm(yprime)
    solver = mysolver()
    y0 = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
    saveat = SaveAt(ts=time)
    solution = diffeqsolve(term, solver, t0=0, t1=100, dt0=0.5, y0=y0, saveat=saveat)
    lm1, lm2, lm3, lm1e, lm2e, lm3e = solution.ys.transpose()

    sig = getsigma([lm1,lm2,lm3,lm1e,lm2e,lm3e], params, useNODE)
    return sig, lm1, lm2, lm3, lm1e, lm2e, lm3e

@partial(jit, static_argnums=(1,))
def triaxial_visco(params, useNODE, time, lamb):
    ipeak = np.argmax(np.abs(np.around(lamb, 3)-1.0)) #around(lm1, 3) evenly rounds lm1 to 3 decimals
    tpeak = time[ipeak]
    lambpeak = lamb[ipeak]

    lambdot = (lambpeak-1.0)/tpeak

    yprime = lambda t, y, args: np.array(yprime_triaxial(y,t,lambdot,tpeak,params,useNODE))
    term = ODETerm(yprime)
    solver = mysolver()
    y0 = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
    saveat = SaveAt(ts=time)
    solution = diffeqsolve(term, solver, t0=0, t1=time[-1], dt0=1.0, y0=y0, saveat=saveat)
    lm1, lm2, lm3, lm1e, lm2e, lm3e = solution.ys.transpose()

    sig = getsigma([lm1,lm2,lm3,lm1e,lm2e,lm3e], params, useNODE)
    return sig, lm1, lm2, lm3, lm1e, lm2e, lm3e

#%% 1. Train Govindjee's model
def loss(params, useNODE, time, lm1, sigma):
    sigma_pr,_,_,_,_,_,_ = triaxial_visco(params, useNODE, time, lm1)
    loss = np.sqrt((sigma_pr[:,0,0]-sigma[:])**2)
    return loss/lm1.shape[0]
loss_vmap = vmap(loss, in_axes=(None, None, 0, 0, 0), out_axes=0)

def batch_loss(params, useNODE, time, lm1, sigma):
    loss = loss_vmap(params, useNODE, time, lm1, sigma)
    loss = np.mean(loss)
    return loss

@partial(jit, static_argnums=(0,1,))
def step(loss_fn, useNODE, i, opt_state, X1_batch, X2_batch, Y_batch):
    params = get_params(opt_state)
    g = grad(loss_fn, argnums=0)(params, useNODE, X1_batch, X2_batch, Y_batch)
    return opt_update(i, g, opt_state)

def train(X1, X2, Y, opt_state, key, nIter = 1000, batch_size=10):
    train_loss = []
    val_loss = []
    for it in range(nIter+1):
        key, subkey = random.split(key)
        i = random.choice(subkey, X1.shape[0], shape=(batch_size,), replace = False)
        i = tuple([i])
        opt_state = step(batch_loss, False, it, opt_state, X1[i], X2[i], Y[i])
        if it % 10 == 0 or it == nIter:
            params = get_params(opt_state)
            train_loss_value = batch_loss(params, False, X1[i], X2[i], Y[i])
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, val_loss 

# Initial values for the material parameters
mu_m = np.array([51.4, -18, 3.86])
alpha_m = np.array([1.8, -2, 7])
K_m = 1.0
tau = 17.5
shear_mod = 1/2*(mu_m[0]*alpha_m[0] + mu_m[1]*alpha_m[1] + mu_m[2]*alpha_m[2])
eta_D = tau*shear_mod
eta_V = tau*(K_m*10000)
mu = 77.77 #=shear_mod
K = 1.0
params = alpha_m, mu_m, K_m, eta_D, eta_V, K, mu

# Training data
with open('training_data/P01.npy','rb') as f:
    time, lamb, sigm = pickle.load(f)
def moving_average(a, n=10):
    ret = onp.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# I know this is a dumb way of doing it. please don't judge me.
print("Training Govindjee's function")
for lr in [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]:
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    params, train_loss, val_loss = train(time, lamb, sigm, opt_state, key, nIter = 100, batch_size=2)
with open('saved/P01_gov_params.npy', 'wb') as f:
    pickle.dump(params, f)
alpha_m, mu_m, K_m, eta_D, eta_V, K, mu = params

#%% 2. Train Psi_EQ and Psi_NEQ

# Stuff common to both Psi_EQ and Psi_NEQ
layers = [1, 5, 5, 1]
def loss(params, lmb, sigma_gt):
    lm1 = lmb[:,0]
    lm2 = lmb[:,1]
    lm3 = lmb[:,2]
    sigma_pr = sigma_split_vmap(lm1, lm2, lm3, params)
    loss1 = np.average((sigma_pr[:,0,0]-sigma_gt[:,0,0])**2) 
    loss2 = np.average((sigma_pr[:,1,1]-sigma_gt[:,1,1])**2)
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
        if (it+1)% 10000 == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it+1, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, val_loss

# Generate inputs
lm = np.linspace(0.9,1.2,20)
lm1, lm2 = np.array(np.meshgrid(lm, lm))
lm1 = lm1.reshape(-1)
lm2 = lm2.reshape(-1)
lm3 = 1/(lm1*lm2)
lmb = np.transpose(np.array([lm1, lm2, lm3]))

#%%% 2.1 Psi_EQ
# Initialize parameters
I1_params = init_params_nobias(layers, key)
I2_params = init_params_nobias(layers, key)
J1_params = init_params_nobias(layers, key)
alpha = 1.0
Psi1_bias = -3.0
Psi2_bias = -3.0
NN_weights = (I1_params, I2_params, J1_params)
Psi_eq_params = (NN_weights, alpha, Psi1_bias, Psi2_bias)

# Generate outputs (Neo Hookean for Psi_EQ)
sigma_gt = []
for i in range(lm1.shape[0]):
    b = np.array([[lm1[i]**2, 0, 0],
                  [0, lm2[i]**2, 0],
                  [0, 0, lm3[i]**2]])
    J = lm1[i]*lm2[i]*lm3[i]
    sigma_gt.append(mu/J*(b-np.eye(3)) + 2*K*(J-1)*np.eye(3))
sigma_gt = np.stack(sigma_gt)

# Train
print("Training Psi_EQ")
opt_init, opt_update, get_params = optimizers.adam(1.e-5)
opt_state = opt_init(Psi_eq_params)
Psi_eq_params, train_loss, val_loss = train(loss, lmb, sigma_gt, opt_state, key, nIter = 100000, batch_size = 10)
with open('saved/P01_Psi_eq_params.npy', 'wb') as f:
    pickle.dump(Psi_eq_params, f)

#%%% 2.2 Psi_NEQ
# Initialize parameters
I1_params = init_params_nobias(layers, key)
I2_params = init_params_nobias(layers, key)
J1_params = init_params_nobias(layers, key)
alpha = 1.0
Psi1_bias = -3.0
Psi2_bias = -3.0
NN_weights = (I1_params, I2_params, J1_params)
Psi_neq_params = (NN_weights, alpha, Psi1_bias, Psi2_bias)

# Generate outputs (Ogden type for Psi_NEQ)
def sigma_NEQ_gov(lm1, lm2, lm3):
    J = lm1*lm2*lm3

    lm1 = J**(-1/3)*lm1
    lm2 = J**(-1/3)*lm2
    lm3 = J**(-1/3)*lm3
    sigma11 = 0
    sigma22 = 0
    sigma33 = 0
    for i in range(3):
        sigma11+= mu_m[i]*lm1**(alpha_m[i]-1)
        sigma22+= mu_m[i]*lm2**(alpha_m[i]-1)
        sigma33+= mu_m[i]*lm3**(alpha_m[i]-1)
    sigma = np.array([[sigma11, 0, 0],
                      [0, sigma22, 0],
                      [0, 0, sigma33]])
    sigma = sigma/J

    p = 1/3*(sigma11 + sigma22 + sigma33)
    sigma = sigma - p

    sigma_vol = K_m/2*(J**2-1)*np.eye(3)
    sigma = sigma + sigma_vol
    return sigma
sigma_NEQ_vmap = vmap(sigma_NEQ_gov, in_axes=(0,0,0), out_axes=0)

sigma_gt = sigma_NEQ_vmap(lm1, lm2, lm3)

print("Training Psi_NEQ")
opt_init, opt_update, get_params = optimizers.adam(1.e-4)
opt_state = opt_init(Psi_neq_params)
Psi_neq_params, train_loss, val_loss = train(loss, lmb, sigma_gt, opt_state, key, nIter = 100000, batch_size = 10)
with open('saved/P01_Psi_neq_params.npy', 'wb') as f:
    pickle.dump(Psi_neq_params, f)

#%% 3. Train Phi
# Generate training data
# Input data and normalization factors
taui = onp.mgrid[-50:10:10j, -50:10:10j, -50:10:10j]
taui = taui.reshape([3,-1]).transpose()
taui = -onp.sort(-taui)

tau1 = taui[:,0]
tau2 = taui[:,1]
tau3 = taui[:,2]
I12     = tau1**2 + tau2**2 + tau3**2 + 2*tau1*tau2 + 2*tau1*tau3 + 2*tau2*tau3
I12m3I2 = tau1**2 + tau2**2 + tau3**2 -   tau1*tau2 -   tau1*tau3 -   tau2*tau3

inp_std1 = onp.std(tau1)
inp_std2 = onp.std(tau1 + tau2)
inp_std3 = onp.std(tau1 + tau2 + tau3)
inp_std4 = onp.std(I12)
inp_std5 = onp.std(I12m3I2)
inp_stds = [inp_std1, inp_std2, inp_std3, inp_std4, inp_std5]

out_std1 = 1.0
out_std2 = 1.0
out_std3 = 1.0
out_std4 = 1/9/175000
out_std5 = 1/3/1360
out_stds = [out_std1, out_std2, out_std3, out_std4, out_std5]

# Output data
dphidtaui = onp.zeros_like(taui)
for i in range(taui.shape[0]):
    dphidtaui[i] = dPhi_gov(taui[i], etad=eta_D, etav=eta_V)

# Initialize NN weights
layers = [1,2,3,1]
NODE1_params = init_params_nobias(layers, key)
NODE2_params = init_params_nobias(layers, key)
NODE3_params = init_params_positivebias(layers, key)
NODE4_params = init_params_positivebias(layers, key)
NODE5_params = init_params_positivebias(layers, key)
Phi_params = [NODE1_params, NODE2_params, NODE3_params, NODE4_params, NODE5_params]

# Train
def loss(params, taui, dphidtaui_gt):
    tau1, tau2, tau3 = taui.transpose()
    dphidtaui_pr = dPhi_vmap([tau1, tau2, tau3], params)
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
    train_loss = []
    for it in range(nIter+1):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch], Y[idx_batch])
        if it % 10000 == 0 or it == nIter:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss

print("Training Phi")
opt_init, opt_update, get_params = optimizers.adam(1.e-4)
opt_state = opt_init(Phi_params)
Phi_params, train_loss = train(loss, taui, dphidtaui, opt_state, key, nIter=200000)

with open('saved/P01_Phi_params.npy', 'wb') as f:
    pickle.dump(Phi_params, f)
with open('saved/P01_Phi_norm_w.npy', 'wb') as f:
    pickle.dump([inp_stds, out_stds], f)

#%% 4. Retrain the entire model with stress-stretch data
params = (Psi_eq_params, Psi_neq_params, Phi_params)
def loss(params, useNODE, time, lm1, sigma1):
    sigma_pr,_,_,_,_,_,_ = triaxial_visco(params, useNODE, time, lm1)
    loss = np.sqrt((sigma_pr[:,0,0]-sigma1[:])**2)
    return loss/lm1.shape[0]
loss_vmap = vmap(loss, in_axes=(None, None, 0, 0, 0), out_axes=0)

def batch_loss(params, useNODE, time, lm1, sigma1):
    loss = loss_vmap(params, useNODE, time, lm1, sigma1)
    loss = np.mean(loss)
    return loss

@partial(jit, static_argnums=(0,1,))
def step(loss_fn, useNODE, i, opt_state, X1_batch, X2_batch, Y1_batch):
    params = get_params(opt_state)
    g = grad(loss_fn, argnums=0)(params, useNODE, X1_batch, X2_batch, Y1_batch)
    return opt_update(i, g, opt_state)

def train(X1, X2, Y1, opt_state, key, nIter = 1000, batch_size=10):
    train_loss = []
    val_loss = []
    for it in range(nIter+1):
        key, subkey = random.split(key)
        i = random.choice(subkey, X1.shape[0], shape=(batch_size,), replace = False)
        i = tuple([i])
        opt_state = step(batch_loss, True, it, opt_state, X1[i], X2[i], Y1[i])
        if it % 100 == 0 or it == nIter:
            params = get_params(opt_state)
            train_loss_value = batch_loss(params, True, X1[i], X2[i], Y1[i])
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, val_loss 

opt_init, opt_update, get_params = optimizers.adam(1.e-5)
opt_state = opt_init(params)

print("Retraining with stress-stretch data")
params, train_loss, val_loss = train(time, lamb, sigm, opt_state, key, nIter = 15000, batch_size=2)
with open('saved/P01_params_retrained.npy', 'wb') as f:
    pickle.dump(params, f)