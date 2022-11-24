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

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
df = pd.read_csv(url, header= None)
X = np.array(df.drop(columns=[0, 1]))
y = np.array(pd.get_dummies(df[1], drop_first= True))
print(y.ravel())

# Instantiate model
svm = SVC(kernel='linear')
lr = LogisticRegression()

# Lasso selector
lasso = Lasso()
model = RFE(lasso)
X_lasso = model.fit_transform(X, y.ravel())

# Boruta selector
rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
X_boruta = boruta_selector.fit_transform(X,  y.ravel())
print(X_boruta.shape)

# Training using Cross Validation with 10 fold
svm_only = cross_val_score(svm, X,  y.ravel(), cv=10) # Train SVM
svm_lasso = cross_val_score(svm, X_lasso,  y.ravel(), cv=10) # Train SVM + LASSO
svm_boruta = cross_val_score(svm, X_boruta,  y.ravel(), cv=10) # Train SVM + Boruta

lr_only = cross_val_score(lr, X,  y.ravel(), cv=10) # Train LR
lr_lasso = cross_val_score(lr, X_lasso,  y.ravel(), cv=10) # Train LR + LASSO
lr_boruta = cross_val_score(lr, X_boruta,  y.ravel(), cv=10) # Train LR + Boruta

print("SVM score: {}".format(svm_only.mean().round(3)))
print("SVM with Lasso: {}".format(svm_lasso.mean().round(3)))
print("SVM with Boruta: {}".format(svm_boruta.mean().round(3)))

print("LR score: {}".format(lr_only.mean().round(3)))
print("LR with Lasso: {}".format(lr_lasso.mean().round(3)))
print("LR with Boruta: {}".format(lr_boruta.mean().round(3)))
