import jax.numpy as np
import jax
from jax import grad, vmap, jit, random, jacrev
from functools import partial
from jax.experimental.ode import odeint
from jax.scipy.optimize import minimize
from jax.lax import scan

@jit
def forward_pass(H, params):
    Ws, b = params
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = np.matmul(H, Ws[i])
        H = np.tanh(H)
    Y = np.matmul(H, Ws[-1]) + np.exp(b) #We want a positive bias
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
# NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

@jit
def NODE_old(y0, params):
    f = lambda y, t: forward_pass(np.array([y]),params) # fake time argument for ODEint
    return odeint(f, y0, np.array([0.0,1.0]))[-1] # integrate between 0 and 1 and return the results at 1

@jit
def sigma_biaxial(lm1, lm2, params):
    lm3 = 1/(lm1*lm2)
    F = np.array([[lm1, 0, 0],
                  [0, lm2, 0],
                  [0, 0, lm3]])
    return sigma(F, params)
sigma_biaxial_vmap = vmap(sigma_biaxial, in_axes = (0, 0, None), out_axes = 0)

@jit
def sigma(F, params):
    C = np.dot(F.T, F)
    PK2 = S(C, params)
    return np.einsum('ij,jk,kl->il', F, PK2, F.T)
sigma_vmap = vmap(sigma, in_axes=(0, None), out_axes=0)
 
@jit
def S(C, params):
    I1_params, I2_params, Iv_params, Iw_params, J1_params, J2_params, J3_params, \
            J4_params, J5_params, J6_params, I_weights, theta, Psi1_bias, Psi2_bias = params
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

    ICs      = [I1,        I2,        Iv,        Iw,        J1,        J2,        J3,        J4,        J5,        J6]
    params   = [I1_params, I2_params, Iv_params, Iw_params, J1_params, J2_params, J3_params, J4_params, J5_params, J6_params]
    derivatives = []
    for IC, Ii_params in zip(ICs, params):
        Psi_i = NODE_nobias(IC, Ii_params)
        derivatives.append(Psi_i)
    for Psi_i in derivatives[2:]:
        Psi_i = np.max(np.array([Psi_i, 0]))
    Psi1, Psi2, Psiv, Psiw, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6 = derivatives

    Psi1 = Psi1 +     a[0]*Phi1 +     a[1]*Phi2 +     a[2]*Phi3 + np.exp(Psi1_bias)
    Psi2 = Psi2 + (1-a[0])*Phi1 +     a[3]*Phi4 +     a[4]*Phi5 + np.exp(Psi2_bias)
    Psiv = Psiv + (1-a[1])*Phi2 + (1-a[3])*Phi4 +     a[5]*Phi6
    Psiw = Psiw + (1-a[2])*Phi3 + (1-a[4])*Phi5 + (1-a[5])*Phi6

    p = -C[2,2]*(2*Psi1 + 2*Psi2*((I1+3) - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
    S = p*Cinv + 2*Psi1*np.eye(3) + 2*Psi2*((I1+3)*np.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0
    return S
S_vmap = vmap(S, in_axes=0, out_axes=0)


@jit
def sigma_split(lm1, lm2, lm3, params): #based on isochoric/volumetric split
    F = np.array([[lm1, 0, 0],
                  [0, lm2, 0],
                  [0, 0, lm3]])
    C = np.dot(F.T, F)
    S = S_split(C, params)
    J = np.linalg.det(F)
    return 1/J*np.einsum('ij,jk,kl->il', F, S, F.T)
sigma_split_vmap = vmap(sigma_split, in_axes=(0, 0, 0, None), out_axes=0)


def S_split(C, params): #The same procedure we use in NNMAT
    NN_weights, alpha, Psi1_bias, Psi2_bias = params
    alpha = 1/(1+np.exp(-alpha))
    J = np.sqrt(np.linalg.det(C))

    I = np.eye(3)
    II = 0.5*(np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
    Cinv = np.linalg.inv(C)
    P = II - 1/3*np.einsum('ij,kl->ijkl', Cinv, C)
    Chat = J**(-2/3)*C
    I1 = np.trace(Chat) #These are actually I1hat and I2hat
    C2 = np.einsum('ij,jk->ik', Chat, Chat)
    I2 = 0.5*(I1**2 - np.trace(C2))

    I1 = I1-3
    I2 = I2**(3/2)-3*np.sqrt(3) #Eq. (2) in Lemma 2.2 of Hartman & Neff 2003. Because I2hat is not polyconvex.
    J1 = alpha*I1+(1-alpha)*I2

    [I1_params, I2_params, J1_params] = NN_weights
    Psi1 = NODE_nobias(I1, I1_params)
    Psi2 = NODE_nobias(I2, I2_params)
    Phi1 = NODE_nobias(J1, J1_params)

    Psi1 = Psi1 +     alpha*Phi1 + np.exp(Psi1_bias)
    Psi2 = Psi2 + (1-alpha)*Phi1 + np.exp(Psi2_bias)

    Shat = 2*Psi1*np.eye(3) + 2*Psi2*((I1+3)*np.eye(3)-Chat)
    Siso = J**(-2/3)*np.einsum('ij,ijkl->kl', Shat, P)
    Siso = J**(-2/3)*np.einsum('ijkl,kl->ij', P, Shat)

    K = 10000
    # p = 2*K*(J-1)
    # Svol = J*p*Cinv
    Svol = K/2*(J**2-1)*Cinv
    S = Siso + Svol
    return S

# # In isochoric deformations this becomes:
# def isosigma(F, params):
#     C = np.dot(F.T, F)
#     B = np.dot(F, F.T)
#     NN_weights, alpha, Psi1_bias, Psi2_bias = params
#     alpha = 1/(1+np.exp(-alpha))
#     J = 1

#     I1 = np.trace(C)
#     C2 = np.einsum('ij,jk->ik', C, C)
#     I2 = 0.5*(I1**2 - np.trace(C2))

#     I1 = I1-3
#     I2 = I2-3
#     J1 = alpha*I1+(1-alpha)*I2

#     [I1_params, I2_params, J1_params] = NN_weights
#     Psi1 = NODE_nobias(I1, I1_params)
#     Psi2 = NODE_nobias(I2, I2_params)
#     Phi1 = NODE_nobias(J1, J1_params)

#     Psi1 = Psi1 +     alpha*Phi1 + np.exp(Psi1_bias)
#     Psi2 = Psi2 + (1-alpha)*Phi1 + np.exp(Psi2_bias)

#     S = 2*Psi1*np.eye(3) + 2*Psi2*((I1+3)*np.eye(3)-C)
#     sigma = 1/J*np.einsum('ij,jk,kl->il', F, S, F.T)
#     sigma = 2/J*Psi1*B + 2/J*Psi2*((I1+3)*B - B**2)
#     return sigma