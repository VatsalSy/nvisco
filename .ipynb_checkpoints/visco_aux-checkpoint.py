import tensorflow as tf
from tensorflow.keras.models import Sequential
import keras
import numpy as np


class nobias_tanh(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(nobias_tanh, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.tanh(tf.matmul(inputs, self.w))
    
class linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,1), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + tf.exp(self.b)
    
class normlayer(keras.layers.Layer):
    def __init__(self):
        super(normlayer, self).__init__()
        self.mean = self.add_weight(shape=(1,1), trainable=False)
        self.sd   = self.add_weight(shape=(1,1), trainable=False)
    
class Ndpdt(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.NN1 = []
        self.NN1.append(nobias_tanh(3,1))
        self.NN1.append(nobias_tanh(3,3))
        self.NN1.append(linear(1,3))
        
        self.NN2 = []
        self.NN2.append(nobias_tanh(3,1))
        self.NN2.append(nobias_tanh(3,3))
        self.NN2.append(linear(1,3))
        
        self.NN3 = []
        self.NN3.append(nobias_tanh(3,1))
        self.NN3.append(nobias_tanh(3,3))
        self.NN3.append(linear(1,3))

        self.NN4 = []
        self.NN4.append(nobias_tanh(3,1))
        self.NN4.append(nobias_tanh(3,3))
        self.NN4.append(linear(1,3))
        
        self.NN5 = []
        self.NN5.append(nobias_tanh(3,1))
        self.NN5.append(nobias_tanh(3,3))
        self.NN5.append(linear(1,3))
        
    def forward_pass(self, layers, inputs):
        x = inputs
        for i in range(len(layers)):
            x = layers[i](x)
        return x
        
    def call(self, inputs):
        evals, evecs = tf.linalg.eigh(inputs) # The eval corresponding to evals[...,i] is evecs[...,:,i]
        #tau3, tau2, tau1 = tf.transpose(evals) # The eval corresponding to tau3 is evecs[...,:,0]
        tau3 = evals[:,0:1]
        tau2 = evals[:,1:2]
        tau1 = evals[:,2:3]
        
        v1v1 = tf.einsum('pi,pj->pij', evecs[:,:,2], evecs[:,:,2])
        v2v2 = tf.einsum('pi,pj->pij', evecs[:,:,1], evecs[:,:,1])
        v3v3 = tf.einsum('pi,pj->pij', evecs[:,:,0], evecs[:,:,0])
        
        NODE1 = tau1
        NODE2 = tau1 + tau2
        NODE3 = tau1 + tau2 + tau3
        NODE4 = tau1**2 + tau2**2 + tau3**2 + 2*tau1*tau2 + 2*tau1*tau3 + 2*tau2*tau3 #I1^2
        NODE5 = tau1**2 + tau2**2 + tau3**2 -   tau1*tau2 -   tau1*tau3 -   tau2*tau3 #I1^2-3I2
        
        dt = 0.02
        for i in range(int(1/dt)):
            NODE1 = NODE1 + dt*self.forward_pass(self.NN1, NODE1)
            NODE2 = NODE2 + dt*self.forward_pass(self.NN2, NODE2)
            NODE3 = NODE3 + dt*self.forward_pass(self.NN3, NODE3)
            NODE4 = NODE4 + dt*self.forward_pass(self.NN4, NODE4)
            NODE5 = NODE5 + dt*self.forward_pass(self.NN5, NODE5)
        
        dphidtau1 = NODE1 + NODE2 + NODE3 + 2*NODE4*(tau1 + tau2 + tau3) + NODE5*(2*tau1 - tau2 - tau3)
        dphidtau2 =         NODE2 + NODE3 + 2*NODE4*(tau1 + tau2 + tau3) + NODE5*(2*tau2 - tau1 - tau3)
        dphidtau3 =                 NODE3 + 2*NODE4*(tau1 + tau2 + tau3) + NODE5*(2*tau3 - tau1 - tau2)
        
        aux1 = tf.einsum('p,pij->pij',dphidtau1[:,0],v1v1)
        aux2 = tf.einsum('p,pij->pij',dphidtau2[:,0],v2v2)
        aux3 = tf.einsum('p,pij->pij',dphidtau3[:,0],v3v3)
        dphidtau = aux1 + aux2 + aux3
        return [dphidtau[:,0,0], dphidtau[:,0,1], dphidtau[:,0,2], dphidtau[:,1,1], dphidtau[:,1,2], dphidtau[:,2,2]]


class Ndpdti(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.NN1 = []
        self.NN1.append(nobias_tanh(3,1))
        self.NN1.append(nobias_tanh(3,3))
        self.NN1.append(linear(1,3))
        
        self.NN2 = []
        self.NN2.append(nobias_tanh(3,1))
        self.NN2.append(nobias_tanh(3,3))
        self.NN2.append(linear(1,3))
        
        self.NN3 = []
        self.NN3.append(nobias_tanh(3,1))
        self.NN3.append(nobias_tanh(3,3))
        self.NN3.append(linear(1,3))

        self.NN4 = []
        self.NN4.append(nobias_tanh(3,1))
        self.NN4.append(nobias_tanh(3,3))
        self.NN4.append(linear(1,3))
        
        self.NN5 = []
        self.NN5.append(nobias_tanh(3,1))
        self.NN5.append(nobias_tanh(3,3))
        self.NN5.append(linear(1,3))
        
        self.inp = normlayer()
        self.out = normlayer()
        
    def forward_pass(self, layers, inputs):
        x = inputs
        for i in range(len(layers)):
            x = layers[i](x)
        return x
        
    def call(self, inputs):
        
        inputs = (inputs-self.inp.mean)/self.inp.sd
        
        tau1 = inputs[:,0:1]
        tau2 = inputs[:,1:2]
        tau3 = inputs[:,2:3]
        
        NODE1 = tau1
        NODE2 = tau1 + tau2
        NODE3 = tau1 + tau2 + tau3
        NODE4 = tau1**2 + tau2**2 + tau3**2 + 2*tau1*tau2 + 2*tau1*tau3 + 2*tau2*tau3 #I1^2
        NODE5 = tau1**2 + tau2**2 + tau3**2 -   tau1*tau2 -   tau1*tau3 -   tau2*tau3 #I1^2-3I2
        
        dt = 0.02
        for i in range(int(1/dt)):
            NODE1 = NODE1 + dt*self.forward_pass(self.NN1, NODE1)
            NODE2 = NODE2 + dt*self.forward_pass(self.NN2, NODE2)
            NODE3 = NODE3 + dt*self.forward_pass(self.NN3, NODE3)
            NODE4 = NODE4 + dt*self.forward_pass(self.NN4, NODE4)
            NODE5 = NODE5 + dt*self.forward_pass(self.NN5, NODE5)
        
        dphidtau1 = NODE1 + NODE2 + NODE3 + 2*NODE4*(tau1 + tau2 + tau3) + NODE5*(2*tau1 - tau2 - tau3)
        dphidtau2 =         NODE2 + NODE3 + 2*NODE4*(tau1 + tau2 + tau3) + NODE5*(2*tau2 - tau1 - tau3)
        dphidtau3 =                 NODE3 + 2*NODE4*(tau1 + tau2 + tau3) + NODE5*(2*tau3 - tau1 - tau2)
        
        dphidtau1 = dphidtau1*self.out.sd + self.out.mean
        dphidtau2 = dphidtau2*self.out.sd + self.out.mean
        dphidtau3 = dphidtau3*self.out.sd + self.out.mean
        return [dphidtau1, dphidtau2, dphidtau3]
    
def dphidtau_gov(tau, etad = 1360, etav = 175000): #This function takes in tau matrix and spits out dphidtau
    trtau = tau[0,0] + tau[1,1] + tau[2,2]
    dphidtau = (2/9/etav - 1/3/etad)*trtau*np.eye(3) + 1/etad*tau #=2*V:tau_neq
    return dphidtau

def dphidtaui_gov(tau_i, etad = 1360, etav = 175000): #This function takes in tau matrix and spits out dphidtau
    tau1, tau2, tau3 = tau_i
    trtau = tau1 + tau2 + tau3
    dphidtau1 = 2*(1/3/etad + 1/9/etav)*trtau - 1/etad*(tau2+tau3)
    dphidtau2 = 2*(1/3/etad + 1/9/etav)*trtau - 1/etad*(tau1+tau3)
    dphidtau3 = 2*(1/3/etad + 1/9/etav)*trtau - 1/etad*(tau1+tau2)
    return [dphidtau1, dphidtau2, dphidtau3]


