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

faces = np.load('faces2.npz')['faces2'].astype(float)
# plt.imshow(faces[:,0].reshape((64,60)))
# plt.show()
avgface = np.average(faces, axis=1)
plt.figure()
# plt.imshow(avgface.reshape((64,60)))
# plt.show()
newdata = np.zeros((3840, 48))
for i in range(48):
  newdata[:,i] = faces[:,i] - avgface
  #plt.imshow(newdata[:,i].reshape((60,64)))
  #plt.show()

w1=np.random.randn(3840,).astype(float);   
w2=np.random.randn(3840,).astype(float)    
num_trials=700;
#eta=0.1/K;
print w1, w2
eta = 0.00001
D = newdata

dw1 = np.zeros((3840,)).astype(float)
dw2 = np.zeros((3840,)).astype(float)
outputs1 = np.zeros((48))
outputs2 = np.zeros((48))
for t in range(num_trials):
  # compute neuron output for all data (can be done as one line)
  for i in range(48):
    #print w1.shape, D[:,i].shape, outputs1.shape
    outputs1[i] = np.dot(w1,D[:,i])
    #print outputs1[i]
    # print w1, D[:,i]
    #print w1, w2
    outputs2[i] = np.dot(w2,D[:,i])
    dw1 = eta * outputs1[i]*(D[:,i] - outputs1[i] * w1)/float(len(outputs1))
    dw2 = eta * outputs2[i] * (D[:,i]- outputs2[i]* w2 - outputs1[i]*w1)/float(len(outputs2))
  #print dw1, dw2
    w1 = w1 + dw1 / np.linalg.norm(w1 + dw1)
    w2 = w2 + dw2 / np.linalg.norm(w1 + dw2)

# eigmovie(np.zeros((60,64)), np.reshape(w2,(60,64)), 1000)
# eigmovie(np.zeros((60,64)), np.reshape(w1,(60,64)), 1000)
# plt.figure()
# plt.imshow(w1.reshape((64,60)))
# plt.show()
# plt.figure()
# plt.imshow(w2.reshape((64,60)))
# plt.show()
#eigmovie(avgface, w1.reshape((64,60)), 100)

plt.imshow(w1.reshape((64,60)))
plt.show()
plt.imshow(w2.reshape((64,60)))
plt.show()

plt.figure()
points1 = [np.dot(faces[:,i].astype(float),w1.astype(float)) for i in range(48)] 
points2 = [np.dot(faces[:,i].astype(float),w2.astype(float)) for i in range(48)]
print points1, points2
plt.scatter(points1,points2)
plt.show()

plt.imshow(faces[:,0].reshape((64,60)))
plt.show()

plt.imshow(avgface.reshape((64,60)))
plt.show()
plt.imshow((avgface + points1[0] * w1 + points2[0] * w2).reshape(64,60).T)
plt.show()

