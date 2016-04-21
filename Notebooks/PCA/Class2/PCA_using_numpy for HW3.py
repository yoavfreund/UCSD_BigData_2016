# Databricks notebook source exported at Wed, 20 Apr 2016 22:30:50 UTC
# MAGIC %md ### Performing PCA on vectors with NaNs
# MAGIC This notebook demonstrates the use of numpy arrays as the content of RDDs

# COMMAND ----------

import numpy as np

def outerProduct(X):
  """Computer outer product and indicate which locations in matrix are undefined"""
  O=np.outer(X,X)
  N=1-np.isnan(O)
  return (O,N)
def sumWithNan(M1,M2):
  """Add two pairs of (matrix,count)"""
  (X1,N1)=M1
  (X2,N2)=M2
  N=N1+N2
  X=np.nansum(np.dstack((X1,X2)),axis=2)
  return (X,N)
  

# COMMAND ----------

# computeCov recieves as input an RDD of np arrays, all of the same length, and computes the covariance matrix for that set of vectors
def computeCov(RDDin):
  RDD=RDDin.map(lambda v:np.insert(v,0,1)) # insert a 1 at the beginning of each vector so that the same 
                                           #calculation also yields the mean vector
  OuterRDD=RDD.map(outerProduct)   # separating the map and the reduce does not matter because of Spark uses lazy execution.
  (S,N)=OuterRDD.reduce(sumWithNan)
  # Unpack result and compute the covariance matrix
  #print 'RDD=',RDD.collect()
  print 'shape of S=',S.shape,'shape of N=',N.shape
  #print 'S=',S
  #print 'N=',N
  E=S[0,1:]
  NE=np.float64(N[0,1:])
  print 'shape of E=',E.shape,'shape of NE=',NE.shape
  Mean=E/NE
  O=S[1:,1:]
  NO=np.float64(N[1:,1:])
  Cov=O/NO - np.outer(Mean,Mean)
  # Output also the diagnal which is the variance for each day
  Var=np.array([Cov[i,i] for i in range(Cov.shape[0])])
  return {'E':E,'NE':NE,'O':O,'NO':NO,'Cov':Cov,'Mean':Mean,'Var':Var}

# COMMAND ----------

# Compute the overall distribution of values and the distribution of the number of nan per year
def find_percentiles(SortedVals,percentile):
  L=len(SortedVals)/percentile
  return SortedVals[L],SortedVals[-L]
  
def computeOverAllDist(rdd0):
  UnDef=np.array(rdd0.map(lambda row:sum(np.isnan(row))).sample(False,0.01).collect())
  flat=rdd0.flatMap(lambda v:list(v)).filter(lambda x: not np.isnan(x)).cache()
  count,S1,S2=flat.map(lambda x: np.float64([1,x,x**2]))\
                  .reduce(lambda x,y: x+y)
  mean=S1/count
  std=np.sqrt(S2/count-mean**2)
  Vals=flat.sample(False,0.0001).collect()
  SortedVals=np.array(sorted(Vals))
  low100,high100=find_percentiles(SortedVals,100)
  low1000,high1000=find_percentiles(SortedVals,1000)
  return {'UnDef':UnDef,\
          'mean':mean,\
          'std':std,\
          'SortedVals':SortedVals,\
          'low100':low100,\
          'high100':high100,\
          'low1000':low100,\
          'high1000':high1000
          }



# COMMAND ----------

# MAGIC %run /Users/yfreund@ucsd.edu/Vault

# COMMAND ----------

AWS_BUCKET_NAME = "mas-dse-public" 
MOUNT_NAME = "NCDC-weather"
OPEN_BUCKET_NAME = "mas-dse-open"
OPEN_MOUNT_NAME = "OPEN-weather"
dbutils.fs.unmount("/mnt/%s" % MOUNT_NAME)
dbutils.fs.unmount("/mnt/%s" % OPEN_MOUNT_NAME)
output_code=dbutils.fs.mount("s3n://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
print 'Mount public status=',output_code
output_code=dbutils.fs.mount("s3n://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, OPEN_BUCKET_NAME), "/mnt/%s" % OPEN_MOUNT_NAME)
print 'Mount open status=',output_code

file_list=dbutils.fs.ls('/mnt/%s/Weather'%MOUNT_NAME)
file_list

# COMMAND ----------

file_list=dbutils.fs.ls('/mnt/%s/Weather'%OPEN_MOUNT_NAME)
file_list

# COMMAND ----------

from numpy import linalg as LA

N=sc.defaultParallelism
print 'Number of executors=',N

STAT={}  # dictionary storing the statistics for each measurement
for meas in measurements:

  Query="SELECT * FROM parquet.`%s`\n\tWHERE measurement = '%s'"%(US_Weather_parquet,meas)
  print Query
  df = sqlContext.sql(Query)
  rdd0=df.map(lambda row:(row['station'],((row['measurement'],row['year']),np.array([np.float64(row[str(i)]) for i in range(1,366)])))).cache()

  rdd1=rdd0.sample(False,1)\
           .map(lambda (key,val): val[1])\
           .cache()\
           .repartition(N)
  print rdd1.count()

  #get basic statistics
  STAT[meas]=computeOverAllDist(rdd1)   # Compute the statistics 
  low1000 = STAT[meas]['low1000']  # unpack the extreme values statistics
  high1000 = STAT[meas]['high1000']

  #clean up table from extreme values and from rows with too many undefinde entries.
  rdd2=rdd1.map(lambda V: np.array([x if (x>low1000-1) and (x<high1000+1) else np.nan for x in V]))
  rdd3=rdd2.filter(lambda row:sum(np.isnan(row))<50)
  Clean_Tables[meas]=rdd3.cache().repartition(N)
  C=Clean_Tables[meas].count()
  print 'for measurement %s, we get %d clean rows'%(meas,C)

  # compute covariance matrix
  OUT=computeCov(Clean_Tables[meas])

  #find PCA decomposition
  eigval,eigvec=LA.eig(OUT['Cov'])

  # collect all of the statistics in STAT[meas]
  STAT[meas]['eigval']=eigval
  STAT[meas]['eigvec']=eigvec
  STAT[meas].update(OUT)

  # print summary of statistics
  print 'the statistics for %s consists of:'%meas
  for key in STAT[meas].keys():
    e=STAT[meas][key]
    if type(e)==list:
      print key,'list',len(e)
    elif type(e)==np.ndarray:
      print key,'ndarray',e.shape
    elif type(e)==np.float64:
      print key,'scalar'
    else:
      print key,'Error type=',type(e)


# COMMAND ----------

STAT_Descriptions=[
('SortedVals', 'Sample of values', 'vector whose length varies between measurements'),
 ('UnDef', 'sample of number of undefs per row', 'vector whose length varies between measurements'),
 ('mean', 'mean value', ()),
 ('std', 'std', ()),
 ('low100', 'bottom 1%', ()),
 ('high100', 'top 1%', ()),
 ('low1000', 'bottom 0.1%', ()),
 ('high1000', 'top 0.1%', ()),
 ('E', 'Sum of values per day', (365,)),
 ('NE', 'count of values per day', (365,)),
 ('Mean', 'E/NE', (365,)),
 ('O', 'Sum of outer products', (365, 365)),
 ('NO', 'counts for outer products', (365, 365)),
 ('Cov', 'O/NO', (365, 365)),
 ('Var', 'The variance per day = diagonal of Cov', (365,)),
 ('eigval', 'PCA eigen-values', (365,)),
 ('eigvec', 'PCA eigen-vectors', (365, 365))
  ]

# COMMAND ----------

from pickle import dumps
dbutils.fs.put("/mnt/OPEN-weather/Weather/STAT.pickle",dumps((STAT,STAT_Descriptions)),True)

# COMMAND ----------

# MAGIC %md ### Sample Stations
# MAGIC Generate a sample of stations, for each one store all available year X measurement pairs.

# COMMAND ----------

import numpy as np
US_Weather_parquet='/mnt/NCDC-weather/Weather/US_Weather.parquet/'
measurements=['TMAX','TMIN','TOBS','SNOW','SNWD','PRCP']
Query="SELECT * FROM parquet.`%s`\n\tWHERE "%US_Weather_parquet+"\n\tor ".join(["measurement='%s'"%m for m in measurements])
print Query
df = sqlContext.sql(Query)

rdd0=df.map(lambda row:(str(row['station']),((str(row['measurement']),row['year']),np.array([np.float64(row[str(i)]) for i in range(1,366)])))).cache().repartition(N)


# COMMAND ----------

rdd0.take(10)

# COMMAND ----------

groups=rdd0.groupByKey().cache()
print 'number of stations=',groups.count()

groups1=groups.sample(False,0.01).collect()
#for group in groups1:
#  print group[0],len(group[1]) #[v[0] for v in group[1]]
groups2=[(g[0],[e for e in g[1]]) for g in groups1]

# COMMAND ----------

from pickle import dumps
dbutils.fs.put("/mnt/OPEN-weather/Weather/SampleStations_copy.pickle",dumps(groups2),True)

# COMMAND ----------

