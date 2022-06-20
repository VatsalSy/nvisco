import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
font = {'size'   : 16}
matplotlib.rc('font', **font)
import time
import pickle
from visco_aux import Ndpdti, dphidtaui_gov

taui = np.mgrid[-10000:10000:10j, -10000:10000:10j, -10000:10000:10j]
taui = taui.reshape([3,-1]).transpose()

taui = -np.sort(-taui)
dphidtaui = np.zeros_like(taui)
for i in range(taui.shape[0]):
    dphidtaui[i] = dphidtaui_gov(taui[i])

inputs = taui
outputs = [dphidtaui[:,0], dphidtaui[:,1], dphidtaui[:,2]]

starttime = time.time()
model = Ndpdti()
model.inp.mean = tf.Variable([[np.mean(inputs) ]],dtype=tf.float32)
model.inp.sd   = tf.Variable([[np.std(inputs)  ]],dtype=tf.float32)
model.out.mean = tf.Variable([[np.mean(outputs)]],dtype=tf.float32)
model.out.sd   = tf.Variable([[np.std(outputs) ]],dtype=tf.float32)
model.compile(loss = 'MSE', optimizer = Adam(learning_rate = 0.001))
fit = model.fit(inputs, outputs, epochs = 5000, verbose = 0, workers=8)
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
ax.set(title='Loss: {loss:.3f}'.format(loss = fit.history['loss'][-1]), ylabel='Loss')
fig.savefig('saved/loss.png')
