from __future__ import print_function
import shutil
# $example on$
from numpy import array
import numpy as np
# $example off$
import csv
from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
# $example off$
import scipy.stats
import math
from random import randint
from operator import itemgetter 
import sys
import os
n_clusters = int(sys.argv[2])
index = 0
total_sampled_points = int(sys.argv[3])

def log_likelihood(dat, K, gmm):
    """ marginal over X """
    log_likelihood = 0.0 
    for n in range (len(dat)):
        log_likelihood += np.log(likelihood(dat[n][index], K, gmm))
        print(n)
    return log_likelihood 

def likelihood(x, K, gmm):
    rs = 0.0
    for k in range(K):
        rs += gmm.weights[k]*scipy.stats.norm(gmm.gaussians[i].mu, gmm.gaussians[i].sigma.toArray()).pdf(x)
    # prob_x = gmm.predictSoft([x])
    # rs = np.prod(prob_x)
    return rs


if __name__ == "__main__":
    sc = SparkContext(appName="GaussianMixtureExample")  # SparkContext
    # $example on$
    # Load and parse the data
    data = sc.textFile(sys.argv[1])
    parsedData = data.map(lambda line: array(([float(x) for x in line.strip().split(",")])[index]))

    # Build the model (cluster the data)
    gmm = GaussianMixture.train(parsedData, n_clusters)

    # Save and load model
    if(os.path.isdir('GMMResult')):
      shutil.rmtree('GMMResult')
    gmm.save(sc, "GMMResult")
    sameModel = GaussianMixtureModel.load(sc, "GMMResult")

    # output parameters of model
    for i in range(n_clusters):
        print("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
              "sigma = ", gmm.gaussians[i].sigma.toArray())

    datfull = data.map(lambda line: array(([float(x) for x in line.strip().split(",")])))
    dat = datfull.take(datfull.count())
    D = len(dat[0])
    # print(dat[:][0])
    probabs = []
    cluster = {}
    for j in range(n_clusters):
        cluster[j] = []
    data_size = datfull.count()

    for x in range(data_size):
        # prob_x =  np.zeros(n_clusters)
        # for i in range(n_clusters):
            # prob_x[i] = (scipy.stats.norm(gmm.gaussians[i].mu, gmm.gaussians[i].sigma.toArray()).pdf(dat[x][index]))*(gmm.weights[i])
        prob_x = gmm.predictSoft([dat[x][index]])
        (cluster[np.argmax(prob_x)]).append(x)    
        total = np.abs(prob_x).sum(axis=0)
        prob_x = np.divide(prob_x,total)
        probabs.append(prob_x)
        print(x)

    # save the soft probabs    
    thefile = open('test.txt', 'w')
    for item in probabs:
        thefile.write("%s\n" % item)
    
    # data point numbers for each cluster
    with open('dict.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in cluster.items():
           writer.writerow([key, value])

    # Sort the clusters by their soft probabilities (NOT USED NOW)
    for j in range(n_clusters):
        values = tuple(probabs[x][j] for x in cluster[j])
        # values = probabs[]#corresponding values
        cluster[j] = [x for (y,x) in sorted(zip(values,cluster[j]))]
    samples = {}

    # Sample data points accorinding to cluster points randomly from each
    for i in range(n_clusters):
        samples[i] = {}
        size = int(len(cluster[i]))
        fraction = float(size)/data_size
        num_sampled_points = int(math.floor(total_sampled_points*fraction))
        if (size == 0):
            continue
        indices = np.random.randint(size,size=num_sampled_points)

        data_indices = itemgetter(*indices)(cluster[i])
        for ind in data_indices:
            (samples[i])[ind] = dat[ind]      
        
    # save samples
    with open('samples.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in samples.items():
           writer.writerow([key, value])

    print("Log Likelihood = ")
    log_lik = log_likelihood(dat, n_clusters, gmm)
    print(log_lik) 


    num_free_params = n_clusters*((D*D - D)/2 + 2*D + 1)
    BIC = np.log(data_size)*num_free_params - 2*log_lik     
    
    print(BIC)    
    # with open('dict_sorted.csv', 'wb') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in cluster.items():
    #        writer.writerow([key, value])
               


    sc.stop()
