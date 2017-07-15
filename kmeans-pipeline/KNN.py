from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
from numpy import genfromtxt
X =  genfromtxt(sys.argv[1], delimiter=',')
nbrs = NearestNeighbors(n_neighbors=int(sys.argv[2]), algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X[int(sys.argv[3])],return_distance=True)
if(sys.argv[4]=='a'):
  flag = 0
  for indx,i in enumerate(indices[0]):
   if(i == int(sys.argv[5])):
     print(i)
     print(indx)
     flag = 1
  if(flag == 0):
    print("Not within " + sys.argv[2] + " neighbours")
else:
  lines = tuple(open(sys.argv[5], 'r'))
  for indx,i in enumerate(indices[0]):
    print("Neighbour No. " + str(indx) + " is " + lines[i])
  print(indices)
