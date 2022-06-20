import jax.numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 16}
matplotlib.rc('font', **font)
from NODE_fns import NODE_old as NODE
from jax import grad, random, jit, partial
from jax.experimental import optimizers
from jax.lax import while_loop
dNODE = grad(NODE)
key = random.PRNGKey(0)

@jit
def nvisco(dt, F, C_i_inv, params):
    #Material parameters:
    mu_m = np.array([51.4, -18, 3.86])
    alpha_m = np.array([1.8, -2, 7])
    K_m = 10000
    tau = 17.5
    shear_mod = 1/2*(mu_m[0]*alpha_m[0] + mu_m[1]*alpha_m[1] + mu_m[2]*alpha_m[2])
    eta_D = tau*shear_mod
    eta_V = tau*K_m
    mu = 77.77 #=shear_mod
    K = 10000
    
    
    #Preprocessing
    be_trial = np.dot(F, np.dot(C_i_inv, F.transpose()))
    lamb_e_trial, n_A = np.linalg.eigh(be_trial)
    n_A = np.real(n_A)
    lamb_e_trial = np.sqrt(np.real(lamb_e_trial))
    eps_e_trial = np.log(lamb_e_trial)
    eps_e = eps_e_trial #Initial guess for eps_e

    def iterate(inputs):
        normres, itr, eps_e, eps_e_trial, dt, params = inputs
        mu_m = np.array([51.4, -18, 3.86])
        alpha_m = np.array([1.8, -2, 7])
        K_m = 10000
        tau = 17.5
        shear_mod = 1/2*(mu_m[0]*alpha_m[0] + mu_m[1]*alpha_m[1] + mu_m[2]*alpha_m[2])
        eta_D = tau*shear_mod
        eta_V = tau*K_m
        
        NODE1_params, NODE2_params, NODE3_params, NODE4_params, NODE5_params = params

        lamb_e = np.exp(eps_e)
        Je = lamb_e[0]*lamb_e[1]*lamb_e[2]
        bbar_e = Je**(-2/3)*lamb_e**2 #(54)

        b1 = bbar_e[0]
        b2 = bbar_e[1]
        b3 = bbar_e[2]

        #Calculate K_AB
        ddev11 = 0
        ddev12 = 0
        ddev13 = 0
        ddev22 = 0
        ddev23 = 0
        ddev33 = 0

        for r in range(3):
            e = alpha_m[r]/2
            ddev11 = ddev11 + mu_m[r]*(2*e)*( 4/9*b1**e + 1/9*(b2**e + b3**e)) #(B12)
            ddev22 = ddev22 + mu_m[r]*(2*e)*( 4/9*b2**e + 1/9*(b1**e + b3**e))
            ddev33 = ddev33 + mu_m[r]*(2*e)*( 4/9*b3**e + 1/9*(b1**e + b2**e))

            ddev12 = ddev12 + mu_m[r]*(2*e)*(-2/9*(b1**e + b2**e) + 1/9*b3**e) #(B13)
            ddev13 = ddev13 + mu_m[r]*(2*e)*(-2/9*(b1**e + b3**e) + 1/9*b2**e)
            ddev23 = ddev23 + mu_m[r]*(2*e)*(-2/9*(b2**e + b3**e) + 1/9*b1**e)
        ddev = np.array([[ddev11, ddev12, ddev13],[ddev12, ddev22, ddev23], [ddev13, ddev23, ddev33]])

        lamb_e = np.exp(eps_e)
        Je = lamb_e[0]*lamb_e[1]*lamb_e[2]
        bbar_e = Je**(-2/3)*lamb_e**2 #(54)

        b1 = bbar_e[0]
        b2 = bbar_e[1]
        b3 = bbar_e[2]

        devtau1 = 0
        devtau2 = 0
        devtau3 = 0
        for r in range(3):
            e = alpha_m[r]/2
            devtau1 = devtau1 + mu_m[r]*(2/3*b1**e - 1/3*(b2**e + b3**e)) #(B8)
            devtau2 = devtau2 + mu_m[r]*(2/3*b2**e - 1/3*(b1**e + b3**e))
            devtau3 = devtau3 + mu_m[r]*(2/3*b3**e - 1/3*(b1**e + b2**e))

        devtau = np.array([devtau1, devtau2, devtau3])

        tau_NEQdyadicI = 3*K_m/2*(Je**2-1) #(B8)
        tau_A = devtau + 1/3*tau_NEQdyadicI #(B8)
        tau_3, tau_2, tau_1 = np.sort(tau_A)

        dN1 = dNODE(tau_1, NODE1_params)
        dN2 = dNODE(tau_1 + tau_2, NODE2_params)
        dN3 = dNODE(tau_1 + tau_2 + tau_3, NODE3_params)
        dN4 = dNODE(tau_1**2 + tau_2**2 + tau_3**2 + 2*tau_1*tau_2 + 2*tau_1*tau_3 + 2*tau_2*tau_3, NODE4_params)
        dN5 = dNODE(tau_1**2 + tau_2**2 + tau_3**2 -   tau_1*tau_2 -   tau_1*tau_3 -   tau_2*tau_3, NODE5_params)

        d2phid11 = dN1 + dN2 + dN3 + 2*dN4 + 2*dN5 #d^2phi/dtau1 dtau1
        d2phid22 =       dN2 + dN3 + 2*dN4 + 2*dN5
        d2phid33 =             dN3 + 2*dN4 + 2*dN5

        d2phid12 =       dN2 + dN3 + 2*dN4 - dN5
        d2phid13 =             dN3 + 2*dN4 - dN5
        d2phid23 =             dN3 + 2*dN4 - dN5

        d2phid2tau = np.array([[d2phid11, d2phid12, d2phid13], [d2phid12, d2phid22, d2phid23], [d2phid13, d2phid23, d2phid33]])

        dtaui_depsej = ddev + K_m*Je**2
        dtaui_depsej = dtaui_depsej[(-tau_A).argsort()] #-tau_A.argsort sorts descending order which is what I need.

        K_AB = np.eye(3) + dt*np.dot(d2phid2tau, dtaui_depsej)

        K_AB_inv = np.linalg.inv(K_AB)

        tau_NEQdyadicI = 3/2*K_m*(Je**2-1) #(B8)

        res = eps_e + dt*(1/2/eta_D*devtau + 1/9/eta_V*tau_NEQdyadicI*np.ones(3))-eps_e_trial #(60)
        deps_e = np.dot(K_AB_inv, -res)
        eps_e = eps_e + deps_e
        normres = np.linalg.norm(res)
        itr+= 1
        return [normres, itr, eps_e, eps_e_trial, dt, params]
        
    #Neuton Raphson
    normres = 1.0
    itr = 0
    itermax = 20
    cond_fun = lambda x: np.sign(x[0]-1.e-6) + np.sign(itermax - x[1]) > 0
    inps = while_loop(cond_fun, iterate, [normres,itr, eps_e, eps_e_trial, dt, params])
    normres, itr, eps_e, eps_e_trial, dt, params = inps
    
    #Now that the iterations have converged, calculate stress
    lamb_e = np.exp(eps_e)
    Je = lamb_e[0]*lamb_e[1]*lamb_e[2]
    bbar_e = Je**(-2/3)*lamb_e**2 #(54)

    b1 = bbar_e[0]
    b2 = bbar_e[1]
    b3 = bbar_e[2]

    devtau1 = 0
    devtau2 = 0
    devtau3 = 0
    for r in range(3):
        e = alpha_m[r]/2
        devtau1 = devtau1 + mu_m[r]*(2/3*b1**e - 1/3*(b2**e + b3**e)) #(B8)
        devtau2 = devtau2 + mu_m[r]*(2/3*b2**e - 1/3*(b1**e + b3**e))
        devtau3 = devtau3 + mu_m[r]*(2/3*b3**e - 1/3*(b1**e + b2**e))

    devtau = np.array([devtau1, devtau2, devtau3])

    tau_NEQdyadicI = 3*K_m/2*(Je**2-1) #(B8)
    tau_A = devtau + 1/3*tau_NEQdyadicI #(B8)
    tau_NEQ = tau_A[0]*np.outer(n_A[:,0], n_A[:,0]) + tau_A[1]*np.outer(n_A[:,1], n_A[:,1]) + tau_A[2]*np.outer(n_A[:,2], n_A[:,2]) #(58)
    b = np.dot(F,F.transpose())
    J = np.linalg.det(F)
    sigma_EQ = mu/J*(b-np.eye(3)) + 2*K*(J-1)*np.eye(3) #neo Hookean material
    sigma = 1/Je*tau_NEQ + sigma_EQ #(7)
    
    #Post processing
    be = np.einsum('i,ji,ki->jk', lamb_e**2, n_A, n_A)
    F_inv = np.linalg.inv(F)
    C_i_inv_new = np.dot(F_inv, np.dot(be, F_inv.transpose()))
    return sigma, C_i_inv_new, lamb_e


    # Biaxial tension in plane stress
@jit
def gov_biaxial(eps_x, eps_y, params, dt=1):
    nsteps = eps_x.shape[0]

    sigma_x = np.zeros(nsteps)
    sigma_y = np.zeros(nsteps)
    time    = np.zeros(nsteps)

    # initial condition for viscous strains 
    C_i_inv   = np.eye(3)
    for i in range(nsteps):
        sigma_z = 0.0
        normres = 1.0
        itr = 0
        itermax = 20
        eps_z = 0.0
        def iterate(inps):
            normres, itr, eps_x, eps_y, params, dt, eps_z, sigma_z, C_i_inv = inps

            # guess for F
            F = np.array([[1+eps_x[i], 0, 0], [0, 1+eps_y[i], 0], [0, 0, 1+eps_z]])
            sigma, C_i_inv_new, lamb_e = nvisco(dt, F, C_i_inv, params)
            res = sigma[2,2]-sigma_z

            # calculate dres with NR 
            F_pz = np.array([[1+eps_x[i], 0, 0], [0, 1+eps_y[i], 0], [0, 0, 1+eps_z+1e-6]])
            sigma_pz, aux, aux2 = nvisco(dt, F_pz, C_i_inv, params)
            dres = (sigma_pz[2,2]-sigma[2,2])/1e-6
            deps = -res/dres
            eps_z += deps
            normres = np.linalg.norm(res)
            itr+=1 
            return [normres, itr, eps_x, eps_y, params, dt, eps_z, sigma_z, C_i_inv]
        
        cond_fun = lambda x: np.sign(x[0]-1.e-6) + np.sign(itermax - x[1]) > 0
        inps = while_loop(cond_fun, iterate, [normres,itr, eps_x, eps_y, params, dt, eps_z, sigma_z, C_i_inv])
        normres, itr, eps_x, eps_y, params, dt, eps_z, sigma_z, C_i_inv = inps
        
        # update the internal variable at end of iterations 
        F = np.array([[1+eps_x[i], 0, 0], [0, 1+eps_y[i], 0], [0, 0, 1+eps_z]])
        sigma, C_i_inv_new, lamb_e = nvisco(dt, F, C_i_inv, params)
        C_i_inv = C_i_inv_new
        sigma_x.at[i].set(sigma[0,0])
        sigma_y.at[i].set(sigma[1,1])
        time.at[i].set(time[i-1]+dt)
    return sigma_x, sigma_y, time


with open('training_data/gov_data.npy','rb') as f:
    _, eps_x, eps_y, sigma_x, sigma_y = np.load(f)


@jit
def loss(params, eps_x, eps_y, sigma_x, sigma_y):
    sigma_x_pred, sigma_y_pred, _ = gov_biaxial(eps_x, eps_y, params)
    loss = np.sum((sigma_x_pred-sigma_x)**2) + np.sum((sigma_x_pred-sigma_x)**2)
    return loss/eps_x.shape[0]

def init_params(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws

layers = [1, 5, 5, 1]
NODE1_params = init_params(layers, key)
NODE2_params = init_params(layers, key)
NODE3_params = init_params(layers, key)
NODE4_params = init_params(layers, key)
NODE5_params = init_params(layers, key)
params = [NODE1_params, NODE2_params, NODE3_params, NODE4_params, NODE5_params]

l = loss(params, eps_x, eps_y, sigma_x, sigma_y)
print('Loss: ', l)