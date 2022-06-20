import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras
import matplotlib
font = {'size'   : 16}
matplotlib.rc('font', **font)
import time
import timeit
import pickle

def eval_gov(tau): #This function takes in tau matrix and spits out dphidtau
    #The values from Gov. (92) divided by 1000
    etad = 1.36
    etav = 175.0 
    
    trtau = tau[0,0] + tau[1,1] + tau[2,2]
    dphidtau = (2/9/etav - 1/3/etad)*trtau*np.eye(3) + 1/etad*tau #=2*V:tau_neq
    return dphidtau

n = 100
tau = np.random.normal(size=[n,3,3])
tau = tau + np.transpose(tau, axes=[0,2,1])
dphidtau = np.zeros_like(tau)
for i in range(n):
    dphidtau[i] = eval_gov(tau[i])

#inputs  =      tau.reshape((n,-1))[:,[0,1,2,4,5,8]]# Only take the 6 independent entries in tau
inputs = tau
dphidtau = dphidtau.reshape((n,-1))
outputs = [dphidtau[:,0], dphidtau[:,1], dphidtau[:,2], dphidtau[:,4], dphidtau[:,5], dphidtau[:,8]]


class nobias_tanh(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(nobias_tanh, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.tanh(tf.matmul(inputs, self.w))
    
class nobias_lin(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(nobias_lin, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,1), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + tf.exp(self.b)

    
class custom_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.NN1 = []
        self.NN1.append(nobias_tanh(3,1))
        self.NN1.append(nobias_tanh(3,3))
        self.NN1.append(nobias_lin (1,3))
        
        self.NN2 = []
        self.NN2.append(nobias_tanh(3,1))
        self.NN2.append(nobias_tanh(3,3))
        self.NN2.append(nobias_lin (1,3))
        
        self.NN3 = []
        self.NN3.append(nobias_tanh(3,1))
        self.NN3.append(nobias_tanh(3,3))
        self.NN3.append(nobias_lin (1,3))

        self.NN4 = []
        self.NN4.append(nobias_tanh(3,1))
        self.NN4.append(nobias_tanh(3,3))
        self.NN4.append(nobias_lin (1,3))
        
        self.NN5 = []
        self.NN5.append(nobias_tanh(3,1))
        self.NN5.append(nobias_tanh(3,3))
        self.NN5.append(nobias_lin (1,3))
        
        self.NN6 = []
        self.NN6.append(nobias_tanh(3,1))
        self.NN6.append(nobias_tanh(3,3))
        self.NN6.append(nobias_lin (1,3))
        
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
        NODE4 = -tau3
        NODE5 = tau1**2 + tau2**2 + tau3**2 + 2*tau1*tau2 + 2*tau1*tau3 + 2*tau2*tau3 #I1^2
        NODE6 = tau1**2 + tau2**2 + tau3**2 - tau1*tau2 - tau1*tau3 - tau2*tau3 #I1^2-3I2
        
        dt = 0.02
        for i in range(int(1/dt)):
            NODE1 = NODE1 + dt*self.forward_pass(self.NN1, NODE1)
            NODE2 = NODE2 + dt*self.forward_pass(self.NN2, NODE2)
            NODE3 = NODE3 + dt*self.forward_pass(self.NN3, NODE3)
            NODE4 = NODE4 + dt*self.forward_pass(self.NN4, NODE4)
            NODE5 = NODE5 + dt*self.forward_pass(self.NN5, NODE5)
            NODE6 = NODE6 + dt*self.forward_pass(self.NN6, NODE6)
        
        dphidtau1 = NODE1 + NODE2 + NODE3         + 2*NODE5*(tau1 + tau2 + tau3) + NODE6*(2*tau1 - tau2 - tau3)
        dphidtau2 =         NODE2 + NODE3         + 2*NODE5*(tau1 + tau2 + tau3) + NODE6*(2*tau2 - tau1 - tau3)
        dphidtau3 =                 NODE3 - NODE4 + 2*NODE5*(tau1 + tau2 + tau3) + NODE6*(2*tau3 - tau1 - tau2)
        
        aux1 = tf.einsum('p,pij->pij',dphidtau1[:,0],v1v1)
        aux2 = tf.einsum('p,pij->pij',dphidtau2[:,0],v2v2)
        aux3 = tf.einsum('p,pij->pij',dphidtau3[:,0],v3v3)
        dphidtau = aux1 + aux2 + aux3
        return [dphidtau[:,0,0], dphidtau[:,0,1], dphidtau[:,0,2], dphidtau[:,1,1], dphidtau[:,1,2], dphidtau[:,2,2]]


starttime = timeit.default_timer()
model = custom_model()

learning_rate = 0.005
model.compile(loss = 'MSE', optimizer = Adam(learning_rate = learning_rate))#, run_eagerly=True)
fit = model.fit(inputs, outputs, epochs = 4000, batch_size = 5, verbose = 0, workers=8)

trtime = time.strftime('%H:%M:%S', time.gmtime(timeit.default_timer()-starttime))
print('Training time: ', trtime)

weights = model.get_weights()
with open('weights.pickle', 'wb') as f:
    pickle.dump(weights,f)

with open('io.pickle', 'wb') as f:
    pickle.dump([inputs, outputs], f)

fig, ax = plt.subplots()
ax.plot(fit.history['loss'])
ax.set_yscale('log')
ax.set(title='Loss: {loss:.3f}'.format(loss = fit.history['loss'][-1]), ylabel='log(loss)')
fig.savefig('loss.png')
