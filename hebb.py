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
w1=np.random.randn(N,1);   
w2=np.random.randn(N,1)     
#print w.shape

# plot weight vector
h=plt.plot([0,w1[0]],[0,w1[1]],'r',linewidth=2);  
g=plt.plot([0,w2[0]],[0,w2[1]],'g',linewidth=2);  

  
num_trials=100;
eta=0.1/K;
D = np.hstack((D1,D2))
dw1 = np.zeros((2,1))
dw2 = np.zeros((2,1))
for t in range(num_trials):
	# compute neuron output for all data (can be done as one line)
	outputs1 = (D[0,:]*w1[0]) + (D[1,:]*w1[1])
	for i in range(len(outputs1)):
		dw1[0] += outputs1[i] * D[0,i] - outputs1[i]**2 * w1[0]
		dw1[1] += outputs1[i] * D[1,i] - outputs1[i]**2 * w1[1]
	dw1[0] = dw1[0]/float(len(outputs1))
	dw1[1] = dw1[1]/float(len(outputs1))
	print dw1
	w1 = w1 + dw1 * eta

# for t in range(num_trials):
# 	# compute neuron output for all data (can be done as one line)
# 	outputs2 = (D[0,:]*w2[0]) + (D[1,:]*w2[1])
# 	for i in range(len(outputs1)):
# 		dw2[0] += outputs2[i] * (D[0,i] - outputs2[i]* w2[0])
# 		dw2[1] += outputs2[i] * (D[1,i] - outputs2[i] * w2[1])
# 	dw2[0] = dw2[0]/float(len(outputs2))
# 	dw2[1] = dw2[1]/float(len(outputs2))
# 	print dw2
# 	w2 = w2 + dw2 * eta

	# compute dw: Hebbian learning rule 

	# update weight vector by dw

	h[0].set_data([0,w1[0]], [0,w1[1]]) # replot weight vector
# g[0].set_data([0,w2[0]], [0,w2[1]]) # replot weight vector
	plt.draw()
	#time.sleep(0.25)
plt.show()
  
