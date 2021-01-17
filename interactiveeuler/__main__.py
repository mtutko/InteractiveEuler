"""
This is the module InteractiveEuler.

It is used to create a fun, interactive fluids solver.
"""

import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsView
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLViewWidget
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import mymath
import fluid as fl

class interactiveeuler(QtWidgets.QMainWindow):
    """ main class for InteractiveEuler """
    def __init__(self):
        super(interactiveeuler, self).__init__()
        self.init_ui()

    def init_ui(self):
        uiPath = os.path.join("interactiveeuler","ui","interactiveeuler.ui")
        self.ui = uic.loadUi(uiPath)
        self.ui.setWindowTitle('InteractiveEuler')
        #self.ui.setWindowIcon(QtGui.QIcon('logo.png'))

        # this is just a test that mymath was imported as expected
        a = mymath.my_square_root(324.0)

        self.ui.fluidWindow.setAspectLocked(True)
        img = pg.ImageItem(np.zeros((200,200)))
        self.ui.fluidWindow.addItem(img)
        ## Set initial view bounds
        self.ui.fluidWindow.setRange(QtCore.QRectF(0, 0, 200, 200))

        ## draw with 5x5 brush
        kern = np.array([
            [0.0,  0.0, 0.25, 0.0, 0.0],
            [0.0,  0.25, 0.5, 0.25, 0.0],
            [0.25, 0.5,  1.0, 0.5,  0.25],
            [0.0,  0.25, 0.5, 0.25, 0.0],
            [0.0,  0.0, 0.25, 0.0, 0.0]])
        img.setDrawKernel(kern, mask=kern, center=(2,2), mode='add')
        img.setLevels([0, 1])

        self.ui.viewTemperatureRadio.setChecked(True)
        self.ui.wallSourceRadio.setChecked(True)

        # there's gotta be a better way to do this, but I don't have it now.
        self.ui.viewTemperatureRadio.toggled.connect(lambda:self.radio_state(self.ui.viewTemperatureRadio))
        self.ui.viewPressureRadio.toggled.connect(lambda:self.radio_state(self.ui.viewPressureRadio))
        self.ui.viewVelocityRadio.toggled.connect(lambda:self.radio_state(self.ui.viewVelocityRadio))
        self.ui.wallSourceRadio.toggled.connect(lambda:self.radio_state(self.ui.wallSourceRadio))
        self.ui.pressureSourceRadio.toggled.connect(lambda:self.radio_state(self.ui.pressureSourceRadio))
        self.ui.temperatureSourceRadio.toggled.connect(lambda:self.radio_state(self.ui.temperatureSourceRadio))

        self.ui.show()

    def radio_state(self,b):
        if b.isChecked():
            print(b.text()+" is selected")
        else:
            print(b.text()+" is deselected")


def main():
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('InteractiveEuler')
    interactiveeuler()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()