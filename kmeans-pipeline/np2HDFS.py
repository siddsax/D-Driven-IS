import numpy as np
import sys
from pyspark import SparkContext,StorageLevel
def toCSVLine(data):
  return ','.join(str(d) for d in data)

sc = SparkContext('local[*]')
a = np.load(sys.argv[1])
print("Parallelizing")
rdd = sc.parallelize(a).persist(StorageLevel.MEMORY_AND_DISK)
print("SAVING")
from time import time

lines = rdd.map(toCSVLine)
lines.saveAsTextFile("HDFS"+sys.argv[1])
#rdd.saveAsObjectFile(str(time()))
