import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Average
from tensorflow.keras import Model,Input
import matplotlib
from sklearn import metrics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_recall_curve
import pickle
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"]="7"

def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model

modellist=[x.strip().split('\t')[0] for x in open('modellist.txt')]
#modellist is consisted of individual allele models created in train_model.py
stackX=None


all_models = list()
stackX=None
for i in range(len(modellist)):
    model = load_model(modellist[i])
    print(modellist[i])
    dataset=np.load('63_data/'+str(modellist[i]).split('rrrs')[0]+'train_1.npz')
    X_test, y_test = dataset['test_x'], to_categorical(dataset['test_y'])
    X_train, y_train = dataset['train_x'], to_categorical(dataset['train_y'])
    fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1],model.predict(X_test)[:,1])
    precision,recall,threshold=precision_recall_curve(y_test[:,1],model.predict(X_test)[:,1])
    all_models.append(model)
    modelpred=model.predict(X_test)[:,1]
    df = pd.DataFrame(np.column_stack((modelpred,y_test[:,1])), columns = ['predicted','actual'])
    df.to_csv(modellist[i]+'_test_output.txt', sep="\t", header=True, index=False)
    with open('individual_model_AUC.txt','a') as f:
        f.write(modellist[i]+'\t'+str(metrics.auc(fpr, tpr))+'\n')
    with open('individual_model_F1.txt','a') as g:
        g.write(modellist[i]+'\t'+str(f1_score(y_test[:,1],[round(x) for x in modelpred]))+'\n')
    with open('individual_model_PR.txt','a') as h:
        h.write(modellist[i]+'\t'+str(metrics.auc(recall,precision))+'\n')
    print(model.predict(X_test).shape)
    if stackX is None:
        stackX=model.predict(X_test)
    else:
        stackX = np.dstack((stackX, model.predict(X_test)))
stackX = stackX.reshape((stackX.shape[0], stackX.shape[2]*stackX.shape[1]))
print(stackX.shape)
linmodel = LinearRegression()
linmodel.fit(stackX, y_test[:,1])

p=linmodel.predict(stackX)
lo, hi = min(p), max(p)
pred=([(i - lo) / (hi-lo) for i in p])

linear_model='Linear_model.pkl'
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test[:,1],pred)
precision2,recall2,threshold2=precision_recall_curve(y_test[:,1],pred)

with open(linear_model,'wb') as file:
    pickle.dump(linmodel,file)

