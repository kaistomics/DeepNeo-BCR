import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def testauc(inputfile):
    ans=[]
    pred=[]
    predbi=[]
    for line in open(inputfile):
        temp=line.strip().split('\t')

        ans.append(float(temp[1]))
        pred.append(float(temp[0]))
        if float(temp[0])>0.3:
            predbi.append(1)
        else:
            predbi.append(0)

    fpr,tpr,thresholds=metrics.roc_curve(ans,pred)
    precision, recall, thresholds = precision_recall_curve(ans,predbi)

    print(inputfile,' roc ',round(metrics.auc(fpr,tpr),3))
    print(inputfile,' pr ',auc(recall,precision))
    print(inputfile,' f1 ',f1_score(ans,predbi))
    print(inputfile,' accuracy ',accuracy_score(ans,predbi))
    plt.figure(0).clf()
    plt.plot(fpr,tpr,label="ROC AUC = "+str(round(metrics.auc(fpr,tpr),3)),color='tab:olive')
    #plt.plot(recall,precision,label="PR AUC = "+str(round(metrics.auc(recall,precision),3)),color='tab:olive')
    plt.savefig('AUC_plots.pdf')



testauc('test.tsv')