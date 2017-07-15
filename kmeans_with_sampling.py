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
    sc = SparkContext('local[*]','kmeans')

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    parser.add_argument('Max_n_components', type=int, help='max num_of_clusters')
    parser.add_argument('num_of_sampled_points', type=int,help='num_of_sampled_points')
    parser.add_argument('n_iter', type=int,help='num_of_iterations')
    parser.add_argument('--ct', type=float, default=1e-3,help='convergence_threshold')
    args = parser.parse_args()
    input_file = args.input_file

    #pkl_file = open(input_file,'rb')
    print("data being loaded.....")
    #final_embed = pickle.load(pkl_file)
    #pkl_file.close()
    data =(sc.textFile(input_file).map(lambda s: np.fromstring(s, dtype=np.float64, sep=",")))#.cache()#sc.parallelize(dat,5000) 
    #data = data_98.repartition(500)
    print("data loaded!")
    D = 128#len(dat[0])
    #da = final_embed#np.load(input_file)#data.take(data_size)
    data_size = data.count()
    #data_size = len(da)
    #print("data being parallelized")
    #data = sc.parallelize(da)
    #print("data parallelized")
    max_n_clusters = args.Max_n_components
    filname = "likelihood_" + str(max_n_clusters)
    if(os.path.exists(filname)):
        os.remove(filname)
    print("Data being unloaded in numpy array")
    print("data unloaded")
    myfile = open(filname,"a")
    z = 10
    while z*100 < max_n_clusters+1:
        n_clusters = z*100
        print(n_clusters)
        print("THE VALUE OF NUMBER OF CLUSTERS IS ABOVE")
        model = KMeans.train(data, n_clusters, initializationMode="k-means||",seed=50, initializationSteps=5, epsilon=1e-3,maxIterations=10000) 
        wssse = model.computeCost(data)
        print("Within Set Sum of Squared Errors = " + str(wssse))

        #  Shows the result.
        centers = model.clusterCenters
        # print("Cluster Centers: ")
        # for center in centers:
            # print(center)
    #     responsibility_matrix, cluster_labels, loglikelihood, cluster_probability = GMMModel.resultPredict(
    #         model, data)

    #     responsibility_matrix_a = responsibility_matrix.take(data_size)
    #     cluster_probability_a = cluster_probability.take(data_size)
    #     cluster_labels_a = cluster_labels.take(data_size)
    # ################SAMPLING##################################################
        total_sampled_points = args.num_of_sampled_points
        cluster = {}
        samples = {}
        da = data.collect()   
        for j in range(n_clusters):
            cluster[j] = []

        
        #print(centers[0])
        for x in range(data_size):
            # (cluster[model.predict(data.zipWithIndex().filter(lambda y : x == y[1]).map(lambda g : g[0]).first())]).append(x)
            (cluster[model.predict(da[x])]).append(x)
        print("dividing to clusters done")
        # Sort the clusters by their soft probabilities (NOT USED NOW)
        for j in range(n_clusters):
            print(j)
            # values = tuple(1.0/(np.linalg.norm(centers[j]-(data.zipWithIndex().filter(lambda y : x == y[1]).map(lambda g : g[0]).first()))+.1) for x in cluster[j])
            values = tuple(np.linalg.norm(centers[j]-(da[x])) for x in cluster[j])
            
            # values = probabs[]#corresponding values
            cluster[j] = [x for (y, x) in sorted(zip(values, cluster[j]))]
        print("dividing sorting done")
        for i in range(n_clusters):
                    samples[i] = {}
                    size = int(len(cluster[i]))
                    fraction = float(size) / data_size
                    num_sampled_points = int(math.floor(total_sampled_points * fraction))
                    if (size == 0):
                        continue

                    indices = np.random.randint(size, size=num_sampled_points)
                    if(num_sampled_points == 0):
                        continue
                    data_indices = itemgetter(*indices)(cluster[i]) 
                    if isinstance(data_indices, int):
                        data_indices = [data_indices]

                    for ind in data_indices:
                        (samples[i])[ind] = dat[ind]

        # save samples
        name = 'samples.csv_' + str(n_clusters)
        with open(name, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in samples.items():
               writer.writerow([key, value])

        # print("Log Likelihood = ")
        # print(loglikelihood)
        # print("BIC = ")
        # l=1.0
        # num_free_params = n_clusters*(2*D + 1)
        # BIC = -2*loglikelihood + .1*num_free_params*np.log(data_size)      
        
        # print(BIC)
        myfile.write( str(n_clusters) + "," + str(wssse) + '\n')
        name = 'cluster_points_sorted.csv_' + str(n_clusters)
        with open(name, 'wb') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in cluster.items():
                   writer.writerow([key, value])
        input_file = 'result_' + str(n_clusters)
        
        if (os.path.isdir(input_file)):
            shutil.rmtree(input_file)
        
        means_file = input_file.split(".")[0]+"/means"
        sc.parallelize(centers, 1).saveAsTextFile(means_file)

        # covar_file = input_file.split(".")[0]+"/covars"
        # sc.parallelize(model.Covars, 1).saveAsTextFile(covar_file)

        # responsbilities = input_file.split(".")[0]+"/responsbilities"
        # responsibility_matrix.coalesce(1).saveAsTextFile(responsbilities)

        # cluster_file = input_file.split(".")[0]+"/clusters"
        # cluster_labels.coalesce(1).saveAsTextFile(cluster_file)
        #os.remove('temp.csv')
        z = z + 1
    sc.stop()

