import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import sys
import pickle
import time
start_time = time.time()
import sys, os
from tensorflow.keras.models import load_model
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
mode=sys.argv[2]

#inputfile = sys.argv[1]

calpha = pd.read_csv('Data/Calpha.txt', sep="\t", header=0, index_col=0)
weight=[]
t=[]
hla = pd.read_csv('Data/BCR_IMGT_multiple_images.txt', sep="\t", header=None, index_col=0)
weight={}
for line in open('Data/Cell_SHM_weight_AA.txt'):
    temp=line.strip().split('\t')
    weight[temp[0]]=temp[1:]

dic_calpha={}
for i in range(calpha.shape[0]):
    for j in range(calpha.shape[1]):
        dic_calpha[(calpha.columns[i], calpha.index[j])] = calpha.iloc[i,j]

with open('Data/Linear_model_'+mode+'.pkl','rb') as file2:
    linear_model = pickle.load(file2)


def get_array(pair):
    thistype=str(pair[1]).split('-')[0]
    mhc = hla.loc[pair[1]].values.tolist()[0]
    epitope = pair[0]
    score = []
    try:
        if len(epitope)==16:
            for i in range(len(epitope)):
                for j in range(len(mhc)):
                    if thistype in weight:
                        score.append(dic_calpha[(epitope[i], mhc[j].replace('.','O'))]*(1+float(weight[thistype][j])))
                    else:
                        thistype='All'
                        score.append(dic_calpha[(epitope[i], mhc[j].replace('.','O'))]*(1+float(weight[thistype][j])))
            return np.reshape(score, (16,len(mhc),1))
    except IndexError:
        pass


modellist=[x.strip().split('\t')[0] for x in open('Data/modellist_'+mode+'.txt')]
def load_all_models(n_models):
    all_models = list()
    stackX=None
    stackXMT=None
    for i in range(len(modellist)):
        model = load_model('Data/models/'+modellist[i])
        tcr1 = pd.read_csv(sys.argv[3], sep="\t", header=None)
        tcr1['allele']=modellist[i].split('_rr')[0]
        arr = np.apply_along_axis(get_array, 1, tcr1)
        mtpep=tcr1[0]
        if stackX is None:
            stackX=model.predict_step(arr)
        else:
            stackX = np.dstack((stackX, model.predict_step(arr)))
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[2]*stackX.shape[1]))
    mtpred=linear_model.predict(stackX)
    df = pd.DataFrame(np.column_stack((mtpep,mtpred)), columns = ['Input','BCR_score'])
    df.to_csv(sys.argv[3]+'_'+mode+'_output.txt', sep="\t", header=True, index=False)

load_all_models(modellist)
