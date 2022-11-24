import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

file_path = "X_GSE2034_series100.csv"
df = pd.read_csv(file_path, header= None)
X = np.array(df.drop(columns=100))
y = np.array(df[100])
print(y)

# Instantiate model
svm = SVC(kernel='linear')
lr = LogisticRegression()

# Lasso selector
lasso = Lasso()
model = RFE(lasso)
X_lasso = model.fit_transform(X, y)

# Boruta selector
rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(rfc, n_estimators=50, verbose=2)
X_boruta = boruta_selector.fit_transform(X,  y)
print(X_boruta.shape)

'''
# Using Train_test split method

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)

X_train, X_test, y_train, y_test = train_test_split(X_lasso, y, random_state=0, train_size=0.75)
lr.fit(X_train, y_train)
lr_lasso_pred = lr.predict(X_test)
print(accuracy_score(y_test, lr_lasso_pred))

X_train, X_test, y_train, y_test = train_test_split(X_boruta, y, random_state=0, train_size=0.7)
svm.fit(X_train, y_train)
svm_boruta_pred = svm.predict(X_test)
print(accuracy_score(y_test, svm_boruta_pred))
'''

# Training using Cross Validation with 10 fold
svm_only = cross_val_score(svm, X,  y, cv=3, verbose=2, n_jobs=-1) # Train SVM
svm_lasso = cross_val_score(svm, X_lasso,  y, cv=3, verbose=2, n_jobs=-1) # Train SVM + LASSO
svm_boruta = cross_val_score(svm, X_boruta,  y, cv=3, verbose=2, n_jobs=-1) # Train SVM + Boruta

lr_only = cross_val_score(lr, X,  y.ravel(), cv=3, verbose=2, n_jobs=-1) # Train LR
lr_lasso = cross_val_score(lr, X_lasso,  y, cv=3, verbose=2, n_jobs=-1) # Train LR + LASSO
lr_boruta = cross_val_score(lr, X_boruta,  y, cv=3, verbose=2, n_jobs=-1) # Train LR + Boruta

print("SVM score: {}".format(svm_only.mean().round(3)))
print("SVM with Lasso: {}".format(svm_lasso.mean().round(3)))
print("SVM with Boruta: {}".format(svm_boruta.mean().round(3)))

print("LR score: {}".format(lr_only.mean().round(3)))
print("LR with Lasso: {}".format(lr_lasso.mean().round(3)))
print("LR with Boruta: {}".format(lr_boruta.mean().round(3)))
