import sys
from PyQt5.QtWidgets import *

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

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(450, 250, 600, 350)
        self.setFixedSize(self.size())
        self.setWindowTitle('Main')
        self.main()


    def main(self):

        #LABEL
        label = QLabel('Breast Cancer Classification ', self)
        label.resize(200, 45)
        label.move(200, 20)

        #self.styleChoice = QLabel("Selection: ", self)
        #self.styleChoice.move(50, 150)
        #self.styleChoice2 = QLabel("Classifier: ", self)
        #self.styleChoice2.move(50, 200)

        # BUTTON
        btn = QPushButton('SVM Only', self)
        btn.setEnabled(True)
        btn.resize(150, 45)
        btn.move(30, 80)
        btn.clicked.connect(self.loadSVM)

        btn = QPushButton('SVM + LASSO', self)
        btn.setEnabled(True)
        btn.resize(150, 45)
        btn.move(200, 80)
        btn.clicked.connect(self.loadSVMLASSO)

        btn = QPushButton('SVM + Boruta', self)
        btn.setEnabled(True)
        btn.resize(150, 45)
        btn.move(370, 80)
        btn.clicked.connect(self.loadLRBoruta)

        btn = QPushButton('LR Only', self)
        btn.setEnabled(True)
        btn.resize(150, 45)
        btn.move(30, 160)
        btn.clicked.connect(self.loadLR)

        btn = QPushButton('LR + LASSO', self)
        btn.setEnabled(True)
        btn.resize(150, 45)
        btn.move(200, 160)
        btn.clicked.connect(self.loadLRLASSO)

        btn = QPushButton('LR + Boruta', self)
        btn.setEnabled(True)
        btn.resize(150, 45)
        btn.move(370, 160)
        btn.clicked.connect(self.loadLRBoruta)

    def nextWindow(self):
        self.window = QMainWindow()
        self.ui = Main()
        self.ui.Main()
        self.close()

    def loadSVM(self):
        lasso = Lasso()
        model = RFE(lasso)
        X_lasso = model.fit_transform(X, y)
        print(X_lasso.shape)

        rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
        X_boruta = boruta_selector.fit_transform(X, y)
        print(X_boruta.shape)

        svm_only = cross_val_score(svm, X, y, cv=10)  # Train SVM
        print("SVM score: {}".format(svm_only.mean().round(3)))
        QMessageBox.question(self,'Message - pythonspot.com', "SVM score: {}".format(svm_only.mean().round(3)))

    def loadSVMBoruta(self):
        lasso = Lasso()
        model = RFE(lasso)
        X_lasso = model.fit_transform(X, y)
        print(X_lasso.shape)

        rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
        X_boruta = boruta_selector.fit_transform(X, y)
        print(X_boruta.shape)

        svm_boruta = cross_val_score(svm, X_boruta, y, cv=10) # Train SVM + Boruta
        print("SVM with Boruta: {}".format(svm_boruta.mean().round(3)))
        QMessageBox.question(self, 'Message - pythonspot.com', "SVM with Boruta: {}".format(svm_boruta.mean().round(3)))

    def loadSVMLASSO(self):
        lasso = Lasso()
        model = RFE(lasso)
        X_lasso = model.fit_transform(X, y)
        print(X_lasso.shape)

        rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
        X_boruta = boruta_selector.fit_transform(X, y)
        print(X_boruta.shape)

        svm_lasso = cross_val_score(svm, X_lasso, y, cv=10) # Train SVM + LASSO
        print("SVM with Lasso: {}".format(svm_lasso.mean().round(3)))
        QMessageBox.question(self, 'Message - pythonspot.com', "SVM with Lasso: {}".format(svm_lasso.mean().round(3)))

    def loadLR(self):
        lasso = Lasso()
        model = RFE(lasso)
        X_lasso = model.fit_transform(X, y)
        print(X_lasso.shape)

        rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
        X_boruta = boruta_selector.fit_transform(X, y)
        print(X_boruta.shape)

        lr_only = cross_val_score(lr, X, y, cv=10) # Train LR
        print("LR score: {}".format(lr_only.mean().round(3)))
        QMessageBox.question(self, 'Message - pythonspot.com', "LR score: {}".format(lr_only.mean().round(3)))

    def loadLRLASSO(self):
        lasso = Lasso()
        model = RFE(lasso)
        X_lasso = model.fit_transform(X, y)
        print(X_lasso.shape)

        rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
        X_boruta = boruta_selector.fit_transform(X, y)
        print(X_boruta.shape)

        lr_lasso = cross_val_score(lr, X_lasso, y, cv=10) # Train LR + LASSO
        print("LR with Lasso: {}".format(lr_lasso.mean().round(3)))
        QMessageBox.question(self, 'Message - pythonspot.com', "LR with Lasso: {}".format(lr_lasso.mean().round(3)))

    def loadLRBoruta(self):
        lasso = Lasso()
        model = RFE(lasso)
        X_lasso = model.fit_transform(X, y)
        print(X_lasso.shape)

        rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
        X_boruta = boruta_selector.fit_transform(X, y)
        print(X_boruta.shape)

        lr_boruta = cross_val_score(lr, X_boruta, y, cv=10) # Train LR + Boruta
        print("LR with Boruta: {}".format(lr_boruta.mean().round(3)))
        QMessageBox.question(self, 'Message - pythonspot.com', "LR with Boruta: {}".format(lr_boruta.mean().round(3)))

    def file_open(self, btn):
        name = QFileDialog.getOpenFileName(self, 'Open File')

        #PROGRESS FOR UPLOADING FILE
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.00005
            self.progress.setValue(self.completed)




if __name__ == "__main__":

    stylesheet = '''
    QMainWindow{
        background-color: #F8CCD9;
    }
    QLabel {
        font: bold 14px;
        font-style: arial;
    }
    QPushButton {
        background-color: #EC769A;
        border-style: outset;
        border-width: 2px;
        border-radius: 22px;
        border-color: black;
        font: bold 12px;
        padding: 6px;
    }
    QPushButton:hover {
        background-color: #F199B3;
        border-style: inset;
        font: bold 13px;
    }
    QPushButton:pressed {
        background-color: #E75480;
        border-style: inset;
        font: bold 12px;
    }
    
    QPushButton:disabled {
        background-color: #F3AAC0;
        border-color: #2F4F4F;
        border-style: inset;
    }
    QLineEdit {
        background-color: white;
        border-radius: 10px;
        border: 2px solid black;
        color: white;
    }
    QProgressBar {
        border: 2px solid #708090;
        border-radius: 5px;
        text-align: center;
        font: bold;
    }
    QProgressBar::chunk {
        background-color: #F199B3;
        margin: 0.5px;
        width: 10px;
    }
    '''
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.setStyleSheet(stylesheet)
    ui.show()

    sys.exit(app.exec_())