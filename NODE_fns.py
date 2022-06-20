import jax.numpy as np
import jax
from jax import grad, vmap, jit, random, jacrev#, partial
from functools import partial
from jax.experimental.ode import odeint
from jax.scipy.optimize import minimize
from jax.lax import scan

# @jit
def forward_pass(H, params):
    Ws, b = params
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = np.matmul(H, Ws[i])
        H = np.tanh(H)
    Y = np.matmul(H, Ws[-1]) + b
    return Y
@jit
def forward_pass_nobias(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = np.matmul(H, Ws[i])
        H = np.tanh(H)
        Y = np.matmul(H, Ws[-1])
    return Y

@jit
def NODE_old(y0, params):
    f = lambda y, t: forward_pass(np.array([y]),params) # fake time argument for ODEint
    return odeint(f, y0, np.array([0.0,1.0]))[-1] # integrate between 0 and 1 and return the results at 1

#The same function as NODE_old except using Euler integration
@jit
def NODE(y0, params, steps = 50):
    body_func = lambda y, i: (y + forward_pass(np.array([y]), params)[0], None)
    out, _ = scan(body_func, y0, None, length = steps)
    return out
NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

@jit
def NODE_nobias(y0, params, steps = 50):
    body_func = lambda y, i: (y + forward_pass_nobias(np.array([y]), params)[0], None)
    out, _ = scan(body_func, y0, None, length = steps)
    return out
NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

@jit
def NODE_sigma(F, params):
    C = np.dot(F.T, F)
    S = NODE_S(C, params)
    return np.einsum('ij,jk,kl->il', F, S, F.T)
NODE_sigma_vmap = vmap(NODE_sigma, in_axes=(0, None), out_axes=0)
 
@jit
def NODE_S(C, params):
    I1_params, I2_params, Iv_params, Iw_params, J1_params, J2_params, J3_params, J4_params, J5_params, J6_params, I_weights, theta, Psi1_bias, Psi2_bias = params
    a = 1/(1+np.exp(-I_weights))
    v0 = np.array([ np.cos(theta), np.sin(theta), 0])
    w0 = np.array([-np.sin(theta), np.cos(theta), 0])
    V0 = np.outer(v0, v0)
    W0 = np.outer(w0, w0)
    I1 = np.trace(C)
    C2 = np.einsum('ij,jk->ik', C, C)
    I2 = 0.5*(I1**2 - np.trace(C2))
    Iv = np.einsum('ij,ij',C,V0)
    Iw = np.einsum('ij,ij',C,W0)
    Cinv = np.linalg.inv(C)

    I1 = I1-3
    I2 = I2-3
    Iv = Iv-1
    Iw = Iw-1
    J1 = a[0]*I1+(1-a[0])*I2
    J2 = a[1]*I1+(1-a[1])*Iv
    J3 = a[2]*I1+(1-a[2])*Iw
    J4 = a[3]*I2+(1-a[3])*Iv
    J5 = a[4]*I2+(1-a[4])*Iw
    J6 = a[5]*Iv+(1-a[5])*Iw

    Psi1 = NODE(I1,  I1_params)
    Psi2 = NODE(I2,  I2_params)
    Psiv = NODE(Iv,  Iv_params)
    Psiw = NODE(Iw,  Iw_params)
    Phi1 = NODE(J1,  J1_params)
    Phi2 = NODE(J2,  J2_params)
    Phi3 = NODE(J3,  J3_params)
    Phi4 = NODE(J4,  J4_params)
    Phi5 = NODE(J5,  J5_params)
    Phi6 = NODE(J6,  J6_params)
    
    Psiv = np.max([Psiv, 0])
    Psiw = np.max([Psiw, 0])
    Phi1 = np.max([Phi1, 0])
    Phi2 = np.max([Phi2, 0])
    Phi3 = np.max([Phi3, 0])
    Phi4 = np.max([Phi4, 0])
    Phi5 = np.max([Phi5, 0])
    Phi6 = np.max([Phi6, 0])
    
    Psi1 = Psi1 +     a[0]*Phi1 +     a[1]*Phi2 +     a[2]*Phi3 + np.exp(Psi1_bias)
    Psi2 = Psi2 + (1-a[0])*Phi1 +     a[3]*Phi4 +     a[4]*Phi5 + np.exp(Psi2_bias)
    Psiv = Psiv + (1-a[1])*Phi2 + (1-a[3])*Phi4 +     a[5]*Phi6
    Psiw = Psiw + (1-a[2])*Phi3 + (1-a[4])*Phi5 + (1-a[5])*Phi6

    p = -C[2,2]*(2*Psi1 + 2*Psi2*((I1+3) - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
    S = p*Cinv + 2*Psi1*np.eye(3) + 2*Psi2*((I1+3)*np.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0
    return S
NODE_S_vmap = vmap(NODE_S, in_axes=0, out_axes=0)