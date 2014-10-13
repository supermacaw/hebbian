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

