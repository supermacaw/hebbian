# hebb.py - script to do hebbian learning
import numpy as np
import matplotlib.pyplot as plt
import time
# you must first load data2d.npz
d = np.load('data2d.npz')
D1,D2 = d['D1'],d['D2']

# data array
X=D1;

N,K=X.shape;

# plot data
plt.plot(X[0,:],X[1,:],'.')                     
#axis xy, axis image
#hold on

# initialize weights
w=np.random.randn(N,1);        
print w.shape

# plot weight vector
h=plt.plot([0,w[0]],[0,w[1]],'r',linewidth=2);  
  
num_trials=100;
eta=0.1/K;
D = np.hstack((D1,D2))
dw = np.zeros((2,1))
for t in range(num_trials):
	# compute neuron output for all data (can be done as one line)
	outputs = (D[0,:]*w[0]) + (D[1,:]*w[1])
	for i in range(len(outputs)):
		dw[0] += outputs[i] * D[0,i]
		dw[1] += outputs[i] * D[1,i]
	dw[0] = dw[0]/float(len(outputs))
	dw[1] = dw[1]/float(len(outputs))
	print dw
	w = w + dw
	#print w

	# compute dw: Hebbian learning rule 

	# update weight vector by dw

	h[0].set_data([0,w[0]], [0,w[1]]) # replot weight vector
	plt.draw()
	time.sleep(0.25)
plt.show()
  
