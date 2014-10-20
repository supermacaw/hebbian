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

# w1=np.random.randn(3840,).astype(float);   
# w2=np.random.randn(3840,).astype(float)  

num_trials=400;
#eta=0.1/K;

eta = 0.00001
D = newdata
numvectors = 15
w = np.random.randn(3840,numvectors) 

outputs = np.zeros((48,numvectors)).astype(float)
dw = np.zeros((3840,numvectors)).astype(float)
# dw1 = np.zeros((3840,)).astype(float)
# dw2 = np.zeros((3840,)).astype(float)
# outputs1 = np.zeros((48))
# outputs2 = np.zeros((48))
for t in range(num_trials): 
  # compute neuron output for all data (can be done as one line)
  for i in range(48):
    #print w1.shape, D[:,i].shape, outputs1.shape
    pastoutputtimesweights = []
    for v in range(numvectors):
      outputs[i][v] = np.dot(w[:,v], D[:,i])
      dw[:,v] = eta * outputs[i][v] * (D[:,i] - outputs[i][v]*w[:,v] - sum(pastoutputtimesweights))/float(48)
      pastoutputtimesweights.append(outputs[i][v] * w[:,v])
      #print pastoutputtimesweights
      w[:,v] = w[:,v] + dw[:,v] / np.linalg.norm(w[:,v] + dw[:,v])


  #   outputs1[i] = np.dot(w1,D[:,i])
  #   #print outputs1[i]
  #   # print w1, D[:,i]
  #   #print w1, w2
  #   outputs2[i] = np.dot(w2,D[:,i])
  #   dw1 = eta * outputs1[i]*(D[:,i] - outputs1[i] * w1)/float(len(outputs1))
  #   dw2 = eta * outputs2[i] * (D[:,i]- outputs2[i]* w2 - outputs1[i]*w1)/float(len(outputs2))
  # #print dw1, dw2
  #   w1 = w1 + dw1 / np.linalg.norm(w1 + dw1)
  #   w2 = w2 + dw2 / np.linalg.norm(w1 + dw2)

# plt.imshow(w1.reshape((64,60)))
# plt.show()
# plt.imshow(w2.reshape((64,60)))
# plt.show()
# for v in range(numvectors):
#   plt.imshow(w[:,v].reshape((64,60)))
#   plt.show()

# plt.figure()
# points1 = [np.dot(faces[:,i].astype(float),w1.astype(float)) for i in range(48)] 
# points2 = [np.dot(faces[:,i].astype(float),w2.astype(float)) for i in range(48)]
# print points1, points2
# plt.scatter(points1,points2)
# plt.show()
points = np.zeros((numvectors, 48))
for v in range(numvectors):
  points[v]= [np.dot(faces[:,i], w[:,v]) for i in range(48)]

# plt.imshow(faces[:,0].reshape((64,60)))
# plt.show()

# plt.imshow(avgface.reshape((64,60)).T)
# plt.show()
plt.imshow((avgface + np.dot(w, points).sum(1)).reshape(64,60).T)
plt.show()

