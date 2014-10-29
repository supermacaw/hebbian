# hebb.py - script to do hebbian learning
import numpy as np
import matplotlib.pyplot as plt
import time
# you must first load data2d.npz
d = np.load('genlines.npy')
# data array
X = d
N,K=X.shape;

# plot data

num_trials=500;
#eta=0.1/K;
eta = 0.001
#D = np.hstack((D1,D2))
# D = D2
# dw = np.zeros((4,2))
# outputs = np.zeros((K,4))
# for t in range(num_trials):
# 	# compute neuron output for all data (can be done as one line)
# 	outputs = np.dot(w, D).T
# 	for output in range(len(outputs)):
# 		argmax = outputs[output].argmax(axis=0)
# 		dw = np.zeros((4,2))
# 		dw[argmax] = eta * outputs[output][argmax] * (D[:, output].T - outputs[output][argmax] * w[argmax]) 
# 		w[argmax] = (w[argmax] + dw[argmax] )/ np.linalg.norm(w[argmax]+dw[argmax])
eta = 0.00001
D = np.load('genlines.npy')
numvectors = 16
w = np.random.randn(64,numvectors) 

outputs = np.zeros((100,numvectors)).astype(float)
dw = np.zeros((64,numvectors)).astype(float)
# dw1 = np.zeros((3840,)).astype(float)
# dw2 = np.zeros((3840,)).astype(float)
# outputs1 = np.zeros((48))
# outputs2 = np.zeros((48))
for t in range(num_trials): 
  # compute neuron output for all data (can be done as one line)
  for i in range(100):
    #print w1.shape, D[:,i].shape, outputs1.shape
    pastoutputtimesweights = []
    for v in range(numvectors):
      outputs[i][v] = np.dot(w[:,v], D[:,i])
      dw[:,v] = eta * outputs[i][v] * (D[:,i] - outputs[i][v]*w[:,v] - sum(pastoutputtimesweights))/float(100)
      pastoutputtimesweights.append(outputs[i][v] * w[:,v])
      #print pastoutputtimesweights
      w[:,v] = w[:,v] + dw[:,v] / np.linalg.norm(w[:,v] + dw[:,v])
	
for v in range(numvectors):
	plt.subplot(4,4,v)
	plt.imshow(w[:,v].reshape((8,8)), cmap='Greys')
plt.show()
#print np.dot(w1, w2)
plt.axis('equal')
plt.show()
  
