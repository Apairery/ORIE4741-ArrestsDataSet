import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.rcdefaults()
# https://zhuanlan.zhihu.com/p/35712080
# https://zhuanlan.zhihu.com/p/35699985
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC                     
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import (precision_recall_curve,
                             auc,roc_auc_score,
                             roc_curve,recall_score,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/data_used.csv') # note: all ready under sampled
print(df.shape)
del df['Unnamed: 0']

drop = ['timestop2','timestop3','sp2','CPI2']
for i in drop:
	del df[i]
# print(df.head())
print(df.shape)
feat_labels = df.columns[1:]

# print(df.arstmade.sum()/len(df))

def under_sample(df):
	"""
	Deal with the unbalanced data: choose nagative samples randomly to match the number of positive samples
	neg:pos = 1:1
	"""
	df_pos = df[df.arstmade.isin([1])]
	df = df[df.arstmade.isin([0])].sample(n=len(df_pos),random_state=123,axis=0)
	df = df.append(df_pos)
	df = df.sample(frac=1).reset_index(drop=True)

	X = df.values[:,1:]
	y = df.values[:,0]
	del df
	return X,y

def performance_metrics(model,X_train,X_test,y_train,y_test):
	# performance of training set

	print("training")
	ypred_rf=model.predict(X_train)

	print('confusion_matrix')
	print(confusion_matrix(y_train,ypred_rf))

	# report
	print('classification_report')
	print(classification_report(y_train,ypred_rf))

	print('Accuracy:%f'%(accuracy_score(y_train,ypred_rf)))
	print('Area under the curve:%f'%(roc_auc_score(y_train,ypred_rf)))


	# performance of test set
	print("test")

	ypred_rf=model.predict(X_test)

	print('confusion_matrix')
	print(confusion_matrix(y_test,ypred_rf))

	# report
	print('classification_report')
	print(classification_report(y_test,ypred_rf))

	print('Accuracy:%f'%(accuracy_score(y_test,ypred_rf)))
	print('Area under the curve:%f'%(roc_auc_score(y_test,ypred_rf)))


def RF_importance(X_train,X_test,y_train,y_test):

	rfmodel=RandomForestClassifier()
	rfmodel.fit(X_train,y_train)

	# feature importance

	importances = rfmodel.feature_importances_
	import_dict = {}
	import_dict['feature'] = feat_labels
	import_dict['importances'] = importances
	df_import = pd.DataFrame(import_dict)
	# print(type(importances))
	df_import.to_csv("RF_gini_importances_drop.csv",index=False)

	return rfmodel


def random_forest(df,test_size):
	# X,y = under_sample(df)

	X = df.values[:,1:]
	y = df.values[:,0]

	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=33) 
	rfmodel = RF_importance(X_train,X_test,y_train,y_test)
	performance_metrics(rfmodel,X_train,X_test,y_train,y_test)

# random_forest(df,0.3)
def SVM_k(df,test_size,kernel):
	# X,y = under_sample(df)

	X = df.values[:,1:]
	y = df.values[:,0]

	X_train,X_test0,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=33) 
	scaler = StandardScaler().fit(X_train)

	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test0)
	print(kernel)

	lsvc = SVC(kernel=kernel)
	lsvc.fit(X_train,y_train)                # 进行模型训练
	## performance_metrics(lsvc,X_train,X_test,y_train,y_test)

	y_test = y_test.reshape(len(y_test),1)
	ypred = lsvc.predict(X_test).reshape(len(y_test),1)
	print(X_test0.shape,y_test.shape,ypred.shape)
	Test = np.concatenate((X_test0,y_test,ypred),axis=1)

	columns = list(df.columns)[1:]+[list(df.columns)[0]]+['pred']

	Test = pd.DataFrame(data=Test, columns=columns)
	Test = Test[['sex_F', 'sex_M', 'race_A', 'race_B', 'race_I', 'race_P', 'race_Q', 'race_W', 'race_Z','arstmade','pred']]
	Test.to_csv('test.csv',index=False)
	print(Test.head())

# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 

# SVM_k(df,0.2,'rbf')

# report FPR among sexes and races
df = pd.read_csv('test.csv')
x = df[df.arstmade.isin([1])]
print(x['pred'].mean())

df = df[df.arstmade.isin([0])]
print(df['pred'].mean())
print(len(df))
for i in ['sex_F', 'sex_M', 'race_A', 'race_B', 'race_I', 'race_P', 'race_Q', 'race_W', 'race_Z']:
	print(i)
	fpr = df[(df[i].isin([1]))]['pred'].mean()
	print(fpr)


