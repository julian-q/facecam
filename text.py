import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label = QLabel("Hello, this is some text!", self)
        label.move(10, 10)
        self.setGeometry(100, 100, 300, 150)
        self.setWindowTitle('PyQt5 Window')
        self.show()

app = QApplication(sys.argv)
window = SimpleWindow()
sys.exit(app.exec_())

