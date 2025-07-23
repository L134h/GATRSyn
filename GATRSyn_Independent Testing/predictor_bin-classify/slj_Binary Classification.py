import os, sys
import math
import numpy as np
import pandas as pd
import pickle
#import keras as K
#import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
#from keras.models import load_model


#os.environ["CUDA_VISIBLE_DEVICES"]="1" #specify GPU


#from keras import backend

#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.callbacks import ModelCheckpoint

def preformance(y_test,y_pre):
    t=30
    tp=len(y_test[(y_test>=t)&(y_pre>=t)])
    fp=len(y_test[(y_test<t)&(y_pre>=t)])
    fn=len(y_test[(y_test>=t)&(y_pre<t)])
    tn=len(y_test[(y_test<t)&(y_pre <t)])
    a=tp+fp
    b=fn+tn
    c=tp+fn
    d=fp+tn
    if a == 0.0:
        precision=0
    else:
        precision = tp/a
   # FPR = fp/(d)
    TPR = tp/c
    TNR = tn/d
    ACC = (tp+tn)/(a+b)
    Pe=(a*c+b*d)/(a+b)/(a+b)
    Pa=(tp+tn)/(a+b)
    Kappa=(Pa-Pe)/(1-Pe)
    return (TPR, TNR, ACC, Kappa,precision)
TPR, TNR, ACC,Precesion,Kappa,PR_AUC,ROC_AUC,BACC=[],[],[],[],[],[],[],[]
s=[]
path1='C:\\Users\\Administrator\\Desktop\\two\\new two category\\A375'  ## 输入向量的路径
path2='C:\\Users\\Administrator\\Desktop\\two\\result'  ## 保存结果的路径
name='A375'                       ## 保存的结果中第一列名字
number=3                          ## 循环的次数
for i in range(number):
    print(os.path.join(path1,'A375observation{}.csv'.format(i+1)))##观测向量的名字
    test=pd.read_csv(os.path.join(path1,'A375observation{}.csv'.format(i+1)),header=None)
    y_t=np.array(test)
    pre=pd.read_csv(os.path.join(path1,'A375prediction{}.csv'.format(i+1)),header=None)##预测向量的名字
    y_p=np.array(pre)

    y_test=y_t[y_t!=0]
    y_pre=y_p[y_t!=0]
    # print(y_test[y_test!=0].shape)
    # print(y_pre[y_test!=0].shape)
    #y_probs = model.predict_proba(X_test) #模型的预测得分
    tpr, tnr, acc,kappa,precision0=preformance(y_test,y_pre)
    bacc=(tpr+tnr)/2.0
    # print(tpr, tnr, acc,kappa,bacc)
    y_test[y_test<30]=0
    y_test[y_test>=30]=1
    # print(y_test)

    fpr, tPr, thresholds = metrics.roc_curve(y_test,y_pre)
    roc_auc = auc(fpr, tPr)  #auc为Roc曲线下的面积
    precision, recall, thresholds = metrics.precision_recall_curve(y_test,y_pre)
    pr_auc = metrics.auc(recall, precision)
    # print(TPR, TNR, ACC, Kappa,precision,PR_auc)
    TPR.append(tpr),TNR.append(tnr), ACC.append(acc), Kappa.append(kappa), \
    Precesion.append(precision0), PR_AUC.append(pr_auc),ROC_AUC.append(roc_auc),BACC.append(bacc)
s.append(TPR)
s.append(TNR)
s.append(ACC)
s.append(Kappa)
s.append(PR_AUC)
s.append(Precesion)
s.append(ROC_AUC)
s.append(BACC)
print(s)
l=['TPR','TNR','ACC','KAPPA','PR_AUC','Precision','ROC_AUC','BACC']
with open(os.path.join(path2,'A375.csv'),'a+') as f:##要保存的文件名字
    f.write(' ')
    f.write(',')
    for n in l:
        f.write (str(n))
        f.write (',')
    f.write('\n')

for i in range(number):
    with open(os.path.join(path2,'A375.csv'),'a+') as f:
        f.write (name+'{}'.format(i+1))
        f.write(',')
        for j in range(len(l)):
            f.write(str(s[j][i]))
            f.write(',')
        f.write('\n')

#开始画ROC曲线
# plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([-0.1,1.1])
# plt.ylim([-0.1,1.1])
# plt.xlabel('False Positive Rate') #横坐标是fpr
# plt.ylabel('True Positive Rate')  #纵坐标是tpr
# plt.title('Receiver operating characteristic curves')
# plt.show()
