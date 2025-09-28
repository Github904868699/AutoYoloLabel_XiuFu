import sys
from PyQt5.QtWidgets import QApplication

from GUI.main import LabelerMainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelerMainWindow()
    window.show()
    sys.exit(app.exec_())
