# eigmovie.py - function to create movie along eigendimension
#
# function eigmovie(mu,e,lim)
#
# mu:   mean
# e:    eigenvector
# lim:  limits of variation
import numpy as np
import matplotlib.pyplot as plt
import time

def eigmovie(mu,e,lim):
    szy=60;
    szx=64;
    mu.shape=(szx,szy);
    e.shape=(szx,szy);

    n=50;
    x=lim*np.sin(2*np.pi*np.linspace(0,1,n));

    h=plt.imshow(mu.T)

    for i in range(n):
      h.set_data((mu+x[i]*e).T)
      plt.draw()
      time.sleep(0.1)

faces = np.load('faces2.npz')['faces2']
faceaggregate = np.zeros((60,64))
for i in range(48):
  face = np.reshape(faces[:,i], (60,64))
  faceaggregate += face
avgface = faceaggregate / float(48)
print avgface.shape
avgfacereshaped = np.reshape(avgface,(3840))
newdata = np.zeros((3840,48))

for i in range(48):
  newdata[:,i] = faces[:,i] - avgfacereshaped

print newdata

w1=np.random.randn(3840,1);   
w2=np.random.randn(3840,1)     
num_trials=200;
#eta=0.1/K;
eta = 1
D = newdata
dw1 = np.zeros((3840,1))
dw2 = np.zeros((3840,1))
for t in range(num_trials):
  # compute neuron output for all data (can be done as one line)
  outputs1 = np.dot(D, w1)
  outputs2 = np.dot(D, w2)
  for i in range(len(outputs1)):
    dw1 += np.dot(outputs1[i],D) - outputs1[i]**2 * w1
    dw2 += outputs2[i] * (D - outputs2[i]* w2 - outputs1[i]*w1)
  dw1[0] = dw1[0]/float(len(outputs1))
  dw1[1] = dw1[1]/float(len(outputs1))
  dw2[0] = dw2[0]/float(len(outputs2))
  dw2[1] = dw2[1]/float(len(outputs2))
  print dw1
  w1 = w1 + dw1 * eta
  w2 = w2 + dw2 * eta
