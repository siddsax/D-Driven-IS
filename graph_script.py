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
#f3 = open(sys.argv[2])
#f4 = open(sys.argv[2])
array = [19867, 2718,13940,15812,28870, 3956,21519, 8430,25101,  23298,26856,18056,13247, 5980,26557, 2664,27119,21575]
number = "9953"
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
name = number+"_word_freq"
w = csv.writer(open(name, "w"))
for key, val in mydict.items():
    w.writerow([key, val])
#fp.close()

