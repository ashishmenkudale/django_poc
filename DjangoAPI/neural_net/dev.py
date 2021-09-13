from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import pickle
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline


df = pd.read_csv('bankloan.csv')
df = df.dropna()
#df.isna().any()
df = df.drop('Loan_ID', axis=1)
df['LoanAmount'] = (df['LoanAmount'] *1000).astype(int)
print(Counter(df['Loan_Status'])) #'Y' l/df[ 'Loan_Status ' .size

pre_y = df['Loan_Status']
pre_X = df.drop('Loan_Status', axis =1)
dm_X = pd.get_dummies(pre_X)
dm_y = pre_y.map(dict(Y=1, N=0))
#print(dm_X.head())

smote = SMOTE(ratio='minority')
X1,y = smote.fit_sample(dm_X, dm_y)
sc = MinMaxScaler()
X = sc.fit_transform(X1)
print(Counter(y))

filename = 'minmaxscaler.pkl'
joblib.dump(sc, filename)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, shuffle=True)

classifier = Sequential()
classifier.add(Dense(200, activation='relu' , kernel_initializer='random_normal', input_dim=X_test.shape[1]))
classifier.add(Dense(400, activation='relu' , kernel_initializer='random_normal'))
classifier.add(Dense(4, activation='relu' , kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid' , kernel_initializer='random_normal'))

classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
classifier.fit(X_train, y_train, batch_size=20, epochs=50, verbose=0)
eval_model= classifier.evaluate(X_train, y_train)
print(eval_model)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);
ax.set_xlabel('Predicted');
ax.set_ylabel('Actual');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['No','Yes'])
ax.yaxis.set_ticklabels(['No','Yes'])

#filename = 'loan_model.pkl'
#joblib.dump(classifier, filename)

#predict
sc = joblib.load('minmaxscaler.pkl')
mdl = joblib.load('loan_model.pkl')

X = pd.read_csv('bankloan.csv', nrows= 5)
X = X.drop('Loan_ID', axis=1)
X = X.dropna()
X['LoanAmount'] = (X['LoanAmount'] *1000).astype(int)
print("here", X.head())

X_test = sc.fit_transform(X)
y_pred = mdl.predict(X_test)
y_pred = (y_pred>0.58)
print(y_pred)



# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# cvscores = []
#
# for train, test in kfold.split(X, y):
#     model = Sequential()
#     model.add(Dense(200, input_dim=17, activation='relu'))
#     model.add(Dense(400, activation='relu'))
#     model.add(Dense(4, activation='sigmoid'))
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     model.fit(X[train], y[train], epochs=100, verbose=0)
#
#     scores = model.evaluate(X[test], y[test], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_name[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))