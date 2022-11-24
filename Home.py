from File import *

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Home(QMainWindow):

    def __init__(self):
        super(Home, self).__init__()
        self.setGeometry(450, 250, 600, 350)
        self.setFixedSize(self.size())
        self.setWindowTitle('Main')
        self.home()

    def home(self):
        # LABEL
        label = QLabel('Breast Cancer Classification ', self)
        label.resize(200, 45)
        label.move(200, 20)

        # BUTTON
        btn1 = QPushButton('File', self)
        btn1.clicked.connect(self.fileWindow)
        btn1.resize(150, 45)
        btn1.move(30, 80)

        btn2 = QPushButton('Internet', self)
        btn2.resize(150, 45)
        btn2.move(200, 80)
        btn2.clicked.connect(self.internetWindow)

        btn3 = QPushButton('SKLearn', self)
        btn3.resize(150, 45)
        btn3.move(370, 80)
        btn3.clicked.connect(self.sklearnWindow)

        self.show()

    def fileWindow(self):
        self.window = QMainWindow()
        self.ui = File()
        self.ui.file()
        self.close()

    def internetWindow(self):
        self.ui.internet()
        self.close()

    def sklearnWindow(self):
        self.ui.Sklearn()
        self.close()


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
    ui = Home()
    ui.setStyleSheet(stylesheet)
    ui.show()

    sys.exit(app.exec_())
