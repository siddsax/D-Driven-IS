import sys
import argparse
#from GMMModel import GMMModel
from pyspark import SparkContext, SparkConf
from pyspark.context import SparkContext

from pyspark.mllib.clustering import KMeans
from pyspark.mllib.random import RandomRDDs
import shutil
import pickle
# import tensorflow as tf
# $example on$
from numpy import array
import numpy as np
# $example off$
import csv
# $example on$
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
# $example off$
import scipy.stats
import math
from random import randint
from operator import itemgetter
import os
from numpy import genfromtxt
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors

def parseVector(line):
    return np.array([float(x) for x in line.split(',')])


if __name__ == "__main__":
    """
    Parameters
    ----------
    input_file : path of the file which contains the comma separated integer data points
    n_components : Number of mixture components
    n_iter : Number of EM iterations to perform. Default to 100
    ct : convergence_threshold.Default to 1e-3
    """
    conf = SparkConf().setAppName("GMM")
    sc = SparkContext("local[*]","kmeans")

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    parser.add_argument('n_components', type=int, help='max num_of_clusters')
    #parser.add_argument('num_of_sampled_points', type=int,help='num_of_sampled_points')
    parser.add_argument('n_iter', type=int,help='num_of_iterations')
    parser.add_argument('--ct', type=float, default=1e-3,help='convergence_threshold')
    args = parser.parse_args()
    input_file = args.input_file

    #pkl_file = open(input_file,'rb')
    print("data being loaded.....")
    #final_embed = pickle.load(pkl_file)
    #pkl_file.close()
    #data =sc.textFile(input_file).map(lambda data: Vectors.dense([float(c) for c in data]))#(sc.textFile(input_file).map(lambda s: np.fromstring(s, dtype=np.float64, sep=",")))#.cache()#sc.parallelize(dat,5000) 
    data = sc.textFile(input_file).map(lambda row: map(lambda x: float(x), row.split(',')))
    #data = data_98.repartition(500)
    print("data loaded!")
    temp =  data.take(1)
    D = len(temp)
    data_size = data.count()
    n_clusters = args.n_components
    n_iter = args.n_iter
    #filname = "likelihood_" + str(max_n_clusters)
    #if(os.path.exists(filname)):
       # os.remove(filname)
    #print("Data being unloaded in numpy array")
    #print("data unloaded")
    #myfile = open(filname,"a")
    print("THE VALUE OF NUMBER OF CLUSTERS IS " + str(n_clusters))
    model = KMeans.train(data, n_clusters, initializationMode="k-means||",seed=50, initializationSteps=5, epsilon=1e-3,maxIterations=n_iter) 
    wssse = model.computeCost(data)
    print("Within Set Sum of Squared Errors = " + str(wssse))
    model.save(sc,"Final_Output_Clusters")
    sc.stop()
        #  Shows the result.
