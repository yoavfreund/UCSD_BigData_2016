import numpy as np
from numpy import linalg as LA

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

def computeCov(RDDin):
    """computeCov recieves as input an RDD of np arrays, all of the same length, 
    and computes the covariance matrix for that set of vectors"""
    RDD=RDDin.map(lambda v:np.insert(v,0,1)) # insert a 1 at the beginning of each vector so that the same 
                                           #calculation also yields the mean vector
    OuterRDD=RDD.map(outerProduct)   # separating the map and the reduce does not matter because of Spark uses lazy execution.
    (S,N)=OuterRDD.reduce(sumWithNan)
    # Unpack result and compute the covariance matrix
    # print 'RDD=',RDD.collect()
    # print 'shape of S=',S.shape,'shape of N=',N.shape
    # print 'S=',S
    # print 'N=',N
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

if __name__=="__main__":
    # create synthetic data matrix with j rows and rank k
    
    V=2*(np.random.random([2,10])-0.5)
    data_list=[]
    for i in range(1000):
        f=2*(np.random.random(2)-0.5)
        data_list.append(np.dot(f,V))
    # compute covariance matrix
    RDD=sc.parallelize(data_list)
    OUT=computeCov(RDD)

    #find PCA decomposition
    eigval,eigvec=LA.eig(OUT['Cov'])
    print 'eigval=',eigval
    print 'eigvec=',eigvec