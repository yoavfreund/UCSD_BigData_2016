from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils

import numpy as np
from time import time
from string import split,strip

class Timer:
    """A simple service class to log run time and pretty-print it.
    """
    def __init__(self):
        self.T=[]
    def stamp(self,name):
        self.T.append((name,time()))
    def str(self):
        T=self.T
        return '\n'.join(['%6.2f : %s'%(T[i+1][1]-T[i][1],T[i+1][0]) for i in range(len(T)-1)])

###### Globals
global T,iteration,GR,proposals,Strong_Classifier, feature_no, partition_no, Splits_Table
global Strong_Classifier,global_best_splitter,PS

T=Timer()
feature_no=None                 # Tracks processing time
global_feature_no=None
partition_no=0
iteration=0                     # Boosting iteration
PS=[None]                       # RDD that hold state of boosting process for each partition.
proposals=[]                    # proposed splits for each feature
Strong_Classifier=[]            # Combined weak classifiers
#############################################################

##### Partition fundctions
def Prepare_partition_data_structure(A):

    rows=len(A[1])

    columns=np.empty([feature_no,rows])
    columns[:]=np.NaN
    print 'Prepare_partition_data_structure',feature_no,np.shape(columns)
    
    labels=np.empty(rows)
    labels[:]=np.NaN

    for j in range(rows):
        LP=A[1][j]
        labels[j]=LP.label
        for i in range(feature_no):
            columns[i,j]=LP.features[i]
    return {'index':A[0],\
            'labels':labels,\
            'weights':np.ones(len(labels)),\
            'feature_values':columns}

def Add_weak_learner_matrix(A):
    """ This procedure adds to each partition the matrix that will be 
        used to efficiently find the best weak classifier """

    try:
        feature_no
    except:
        feature_no=global_feature_no.value

    index=A['index']%feature_no
    SP=Splits_Table.value[index]

    Col=A['feature_values'][index,:]

    ### The matrix M is organized as follows: 
    # * There are as many rows as there are thresholds in SP (last one is inf)
    # * There are as many columns as there are examples in this partition.
    # For threshold i, the i'th rw of M is +1 if Col is smaller than the trehold SP[i] and -1 otherwise

    M=np.empty([len(SP),len(Col)])
    M[:]=np.NaN

    for i in range(len(SP)):
        M[i,:]=2*(Col<SP[i])-1

    A['M']=M # add M matrix to the data structure.
    return A


def Find_weak(A):
    """Find the best split for a single feature on a single partition"""

    try:
        feature_no
    except:
        feature_no=global_feature_no.value

    index=A['index']%feature_no
    SP=Splits_Table.value[index]

    M=A['M']
    weights=A['weights']
    weighted_Labels=weights*A['labels']
    SS=np.dot(M,weighted_Labels)/np.sum(weights)
    i_max=np.argmax(np.abs(SS))
    answer={'Feature_index':A['index']%feature_no,\
            'Threshold_index':i_max,\
            'Threshold':SP[i_max],\
            'Correlation':SS[i_max],\
            'SS':SS}
    return answer

# update weights. New splitter is shipped to partition as one of the referenced
# Variables

def update_weights(A):
    """Update the weights of the exammples belonging to this 
    partition according to the new splitter"""
    best_splitter=global_best_splitter
    F_index=best_splitter['Feature_index']
    Thr=best_splitter['Threshold']
    alpha=best_splitter['alpha']
    y_hat=2*(A['feature_values'][F_index,:]<Thr)-1
    y=A['labels']
    weights=A['weights']*exp(-alpha*y_hat*y)
    weights /= sum(weights)
    A['weights']=weights
    return A

def calc_scores(Strong_Classifier,Columns,Lbl):
    
    Scores=np.zeros(len(Lbl))

    for h in Strong_Classifier:
        index=h['Feature_index']
        Thr=h['Threshold']
        alpha=h['alpha']
        y_hat=2*(Columns[index,:]<Thr)-1
        Scores += alpha*y_hat*Lbl
    return Scores

if __name__ == '__main__':
    from os.path import exists
    if not exists('higgs'):
        print "creating directory higgs"
        get_ipython().system(u'mkdir higgs')
    get_ipython().magic(u'cd higgs')
    if not exists('HIGGS.csv'):
        if not exists('HIGGS.csv.gz'):
            print 'downloading HIGGS.csv.gz'
            get_ipython().system(u'curl -O http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')
        print 'decompressing HIGGS.csv.gz --- May take 5-10 minutes'
        get_ipython().system(u'gunzip -f HIGGS.csv.gz')
    get_ipython().system(u'ls -l')

    #copy file to HDFS - when runnnig on AWS cluster
    get_ipython().system(u'/root/ephemeral-hdfs/bin/hdfs dfs -cp file:///mnt/higgs/HIGGS.csv /HIGGS.csv')

def test_globals():
    return globals()


###### Head-Node functions
def init(sc,Data):
    """ Given an RDD with labeled Points, create the RDD of data structures used for boosting
    """

    global T,iteration,GR,proposals,Strong_Classifier, feature_no, partition_no, Splits_Table
    global Strong_Classifier,global_best_splitter

    T=Timer()
    T.stamp('Started')

    X=Data.first()
    feature_no=len(X.features)
#    print 'global_feature_no = sc.broadcast(feature_no)',feature_no
    partition_no=Data.getNumPartitions()
    if partition_no != feature_no:
        Data=Data.repartition(feature_no).cache()
    print 'number of features=',feature_no,'number of partitions=',Data.getNumPartitions()

    # Split data into training and test
    (trainingData,testData)=Data.randomSplit([0.7,0.3])
    print 'Sizes: Data1=%d, trainingData=%d, testData=%d'%      (Data.count(),trainingData.cache().count(),testData.cache().count())
    T.stamp('Split into train and test')
    # Glom each partition into a local array
    G=trainingData.glom()
    GTest=testData.glom()  
    T.stamp('glom')

    # Add an index to each partition to identify it.
    def f(splitIndex, iterator): yield splitIndex,iterator.next()
    GI=G.mapPartitionsWithIndex(f)
    GTI=GTest.mapPartitionsWithIndex(f)
    T.stamp('add partition index')

    return GI

def init2(GI):
    # Prepare the data structure for each partition.
    GR=GI.map(Prepare_partition_data_structure)
    print 'number of elements in GR=', GR.cache().count()
    T.stamp('Prepare_partition_data_structure')

    #compute the split points for each feature
    Splits=find_splits(GR)
    print 'Split points=',Splits
    T.stamp('Compute Split points')

    #broadcast split points
    global Splits_Table
    Splits_Table=sc.broadcast(Splits)
    T.stamp('Broadcast split points')

    # Create matrix for each partition to make finding the weak rules correlation a matter of taking a matrix product

    iteration=0
    global PS
    PS[0]=GR.map(Add_weak_learner_matrix)
    print 'number of partitions in PS=',PS[0].cache().count()
    T.stamp('Add_weak_learner_matrix')

    return PS

def boosting_iteration(k=1):
    """ perform k boosting iterations """
    for i in range(iteration,iteration+k):
        T.stamp('Start main loop %d'%i)

        prop=PS[i].map(Find_weak).collect()
        proposals.append(prop)
        corrs=[p['Correlation'] for p in prop]
        best_splitter_index=np.argmax(np.abs(corrs))
        best_splitter = prop[best_splitter_index]
        Strong_Classifier.append(best_splitter)
        global global_Strong_Classifier
        global_Strong_Classifier=sc.broadcast(Strong_Classifier)
        T.stamp('found best splitter %d'%i)

        corr=best_splitter['Correlation']
        best_splitter['alpha']=0.5*np.log((1+corr)/(1-corr))
        global global_best_splitter
        global_best_splitter = sc.broadcast(best_splitter)
        PS.append(PS[i].map(update_weights))
        T.stamp('Updated Weights %d'%i)
    iteration+=k

def find_splits(GR,number_of_bins=10,debug=False):
    """Compute the split points for each feature to create number_of_bins bins"""
    def find_split_points(A):

        try:
            feature_no
        except:
            feature_no=global_feature_no.value

        j=A['index'] % feature_no
        S=np.sort(A['feature_values'][j,:       ])
        L=len(S) 
        step=L/number_of_bins+2*number_of_bins
        return (j,S[range(0,L,step)])

    global partition_no
    Splits=GR.map(find_split_points).collect()
    max_no=np.array([np.finfo(float).max])

    # Average the split points across the partitions corresponding to the same feature.
    Splits1=[]
    for i in range(feature_no):
        S=Splits[i][1]
        if debug:
            print 'no. ',i,' = ',Splits[i]
        n=1  # number of copies (for averaging)
        j=i+feature_no
        while j<partition_no:
            if debug:
                print 'j=',j
            S+=Splits[j][1]
            if debug:
                print 'no. ',j,' = ',Splits[j]
            n+=1.0
            j+=feature_no
        Splits1.append(np.concatenate([S/n,max_no]))
        if debug:
            print n
            print Splits1[i]
            print '='*60

    return Splits1

