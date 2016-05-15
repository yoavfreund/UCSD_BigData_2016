from pylab import *
def extract_errors(B):
    keys=B.keys()
    keys.sort()
    train=[]; test=[];
    for key in keys:
        train.append(B[key]['train'])
        test.append(B[key]['test'])
    return keys,train,test

def make_figure(DataSets,names,Title='no title',size=(8,6)):
    figure(figsize=size)
    for i in range(len(DataSets)):
        keys,train,test = extract_errors(DataSets[i])
        plot(keys,train,label='train-'+names[i])
        plot(keys,test,label='test-'+names[i])
    title(Title)
    grid()
    legend()