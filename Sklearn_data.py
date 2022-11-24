from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data['data']
y = data['target']

# Instantiate model
svm = SVC(kernel='linear')
lr = LogisticRegression()

# Lasso selector
lasso = Lasso()
model = RFE(lasso)
X_lasso = model.fit_transform(X, y)
print(X_lasso.shape)

# Boruta selector
rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
X_boruta = boruta_selector.fit_transform(X, y)
print(X_boruta.shape)

# Training using Cross Validation with 10 fold
svm_only = cross_val_score(svm, X, y, cv=10) # Train SVM
svm_lasso = cross_val_score(svm, X_lasso, y, cv=10) # Train SVM + LASSO
svm_boruta = cross_val_score(svm, X_boruta, y, cv=10) # Train SVM + Boruta

lr_only = cross_val_score(lr, X, y, cv=10) # Train LR
lr_lasso = cross_val_score(lr, X_lasso, y, cv=10) # Train LR + LASSO
lr_boruta = cross_val_score(lr, X_boruta, y, cv=10) # Train LR + Boruta

print("SVM score: {}".format(svm_only.mean().round(3)))
print("SVM with Lasso: {}".format(svm_lasso.mean().round(3)))
print("SVM with Boruta: {}".format(svm_boruta.mean().round(3)))

print("LR score: {}".format(lr_only.mean().round(3)))
print("LR with Lasso: {}".format(lr_lasso.mean().round(3)))
print("LR with Boruta: {}".format(lr_boruta.mean().round(3)))
