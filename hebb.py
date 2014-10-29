# hebb.py - script to do hebbian learning
import numpy as np
import matplotlib.pyplot as plt
import time
# you must first load data2d.npz
d = np.load('faces2.npz')['faces2']
#D1,D2 = d['D1'],d['D2']

# data array
#X=D2;

N,K=d.shape;

# plot data
#plt.plot(X[0,:],X[1,:],'.')                     
#axis xy, axis image
#hold on

# initialize weights
import random
numvectors = 10
#w = np.random.randn(numvectors,N)  
weightindices = random.sample(range(0,47), numvectors)
w = np.zeros((numvectors,3840))
for i in range(numvectors):
	w[i,:] = d[:,weightindices[i]]
#w = d.random.random((numvectors,3840))

# plot weight vector
# h=plt.plot([0,w[0][0]],[0,w[0][1]],'r',linewidth=2);  
# g=plt.plot([0,w[1][0]],[0,w[1][1]],'g',linewidth=2);  
# i=plt.plot([0,w[2][0]],[0,w[2][1]],'y',linewidth=2);  
# j=plt.plot([0,w[3][0]],[0,w[3][1]],'m',linewidth=2);  
#plt.show()

num_trials=10000;
#eta=0.1/K;
eta = 0.0001
#D = np.hstack((D1,D2))
#D = D2
D = d
dw = np.zeros((numvectors,N))
outputs = np.zeros((K,numvectors))
for t in range(num_trials):
	# compute neuron output for all data (can be done as one line)
	outputs = np.dot(w, D).T
	for output in range(len(outputs)):
		argmax = outputs[output].argmax(axis=0)
		dw = np.zeros((numvectors,N))
		dw[argmax] = eta * outputs[output][argmax] * (D[:, output].T - outputs[output][argmax] * w[argmax]) 
		w[argmax] = (w[argmax] + dw[argmax] )/ np.linalg.norm(w[argmax]+dw[argmax])
	

# h[0].set_data([0,w[0][0]], [0,w[0][1]]) # replot weight vector
# g[0].set_data([0,w[1][0]], [0,w[1][1]]) # replot weight vector
# i[0].set_data([0,w[2][0]], [0,w[2][1]]) # replot weight vector
# j[0].set_data([0,w[3][0]], [0,w[3][1]])
for i in range(numvectors):
	plt.imshow(w[i].reshape((64,60)).T,cmap='Greys')
	plt.show()
	#plt.draw()
	#time.sleep(0.25)
#print np.dot(w1, w2)
#plt.axis('equal')
#plt.show()
  
