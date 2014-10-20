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
      #time.sleep(0.1)
    plt.show()

faces = np.load('faces2.npz')['faces2']
faceaggregate = np.zeros((60,64))
for i in range(48):
  face = np.reshape(faces[:,i], (60,64))
  faceaggregate += face
avgface = faceaggregate / float(48)
avgfacereshaped = np.reshape(avgface,(3840))
newdata = np.zeros((3840,48))

for i in range(48):
  newdata[:,i] = faces[:,i] - avgfacereshaped

w1=np.random.randn(3840,);   
w2=np.random.randn(3840,)     
num_trials=100;
#eta=0.1/K;
print w1, w2
eta = 0.01
D = newdata

dw1 = np.zeros((3840,))
dw2 = np.zeros((3840,))
outputs1 = np.zeros((3840,48))
outputs2 = np.zeros((3840,48))
for t in range(num_trials):
  # compute neuron output for all data (can be done as one line)
  for i in range(48):
    #print w1.shape, D[:,i].shape, outputs1.shape
    outputs1[:,i] = w1*D[:,i]
    print outputs1[i]
    # print w1, D[:,i]
    #print w1, w2
    outputs2[:,i] = w2*D[:,i]
  for i in range(48):
    dw1 += eta * outputs1[:,i]*(D[:,i] - outputs1[:,i] * w1)/float(len(outputs1))
    dw2 += eta * outputs2[:,i] * (D[:,i]- outputs2[:,i]* w2 - outputs1[:,i]*w1)/float(len(outputs2))
  print dw1, dw2
  w1 = w1 + dw1 * eta
  w2 = w2 + dw2 * eta

eigmovie(avgface, np.reshape(w1,(60,64)), 10)
eigmovie(avgface, np.reshape(w2,(60,64)), 10)
print w1 - w2