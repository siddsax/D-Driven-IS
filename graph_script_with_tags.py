from __future__ import print_function
import csv
import collections
import json
import sys
import itertools
import re
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import os

f1 = open(sys.argv[1])
f2 = open(sys.argv[1])
f3 = open(sys.argv[2])
f4 = open(sys.argv[2])
array = [19867, 2718,13940,15812,28870, 3956,21519, 8430,25101,  23298,26856,18056,13247, 5980,26557, 2664,27119,21575]
array.sort()
print(array)
j = 0
mydict = {}
for line in f1:
    words = line.split()
    for word in words:
        if word not in mydict:
            mydict[word] = 0
#print(mydict)  
for i, line in enumerate(f2):
    #print(j)
    if(j==len(array)):
        print(str(len(array))+"wabalab dubb dubb")
        break
    if i == array[j]:
      j = j + 1
      words = line.split()
      for w in words:
        print(w)
        mydict[w] = mydict[w] + 1
tagdict = {}
j = 0
for line in f3:
    words = line.split("|")
    for word in words:
        if word not in tagdict:
             tagdict[word] = 0

for i, line in enumerate(f4):
    #print(j)
    if(j==len(array)):
        print(str(len(array))+"wabalab dubb dubb")
        break
    if i == array[j]:
      j = j + 1
      words = line.split("|")
      for w in words:
        if(w == '<city<' or w == '<city'):
           w = '<city>'
           print("LLLLLLLLLLLL")
        if(w == '<stree>' or w == '<street'):
           w = '<street>'
        tagdict[w] = tagdict[w] + 1
print(j)
#print(array[j])
#print(i)
print(tagdict['<city<'])
print(tagdict['<city'])
#print(tagdict['<stree'])
print(tagdict['<street'])
#del tagdict['\n']
del tagdict['']
del tagdict['<street']
#del tagdict['<stree']
del tagdict['<city']
del tagdict['<city<']

w = csv.writer(open("2651_cluster.csv", "w"))
for key, val in mydict.items():
    w.writerow([key, val])
#fp.close()

w = csv.writer(open("tags_2651_cluster.csv", "w"))
for key, val in tagdict.items():
    w.writerow([key, val])
