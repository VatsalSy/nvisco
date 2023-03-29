import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 16}
matplotlib.rc('font', **font)
import time
import pickle
from visco_aux import phi_5NODEs, dphi_gov

n = 1000
tau = np.random.normal(size=[n,3,3])
tau = tau + np.transpose(tau, axes=[0,2,1])
dphidtau = np.zeros_like(tau)
for i in range(n):
    dphidtau[i] = dphi_gov(tau[i])

#inputs  =      tau.reshape((n,-1))[:,[0,1,2,4,5,8]]# Only take the 6 independent entries in tau
inputs = tau
dphidtau = dphidtau.reshape((n,-1))
outputs = [dphidtau[:,0], dphidtau[:,1], dphidtau[:,2], dphidtau[:,4], dphidtau[:,5], dphidtau[:,8]]

starttime = time.time()
model = phi_5NODEs()
learning_rate = 0.005
model.compile(loss = 'MSE', optimizer = Adam(learning_rate = learning_rate))#, run_eagerly=True)
fit = model.fit(inputs, outputs, epochs = 2000, batch_size = 5, verbose = 0, workers=8)
trtime = time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))
print('Training time: ', trtime)

weights = model.get_weights()
with open('saved/weights.pickle', 'wb') as f:
    pickle.dump(weights,f)

with open('saved/io.pickle', 'wb') as f:
    pickle.dump([inputs, outputs], f)

fig, ax = plt.subplots()
ax.plot(fit.history['loss'])
ax.set_yscale('log')
ax.set(title='Loss: {loss:.3f}'.format(loss = fit.history['loss'][-1]), ylabel='log(loss)')
fig.savefig('saved/loss.png')
