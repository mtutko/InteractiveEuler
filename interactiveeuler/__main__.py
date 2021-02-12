"""
This is the module InteractiveEuler.

It is used to create a fun, interactive fluids solver.
"""

import sys

import numpy as np
from matplotlib import cm

import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets

import mymath
import fluid as fl


N = 5
MATSIZE = (N, N)

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


def get_matplotlib_lut(name):
    # Get the colormap
    colormap = cm.get_cmap(name)  # cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

    return lut


def getDomain():
    dx = 1 / N
    steparr = np.arange(dx/2, 1 + dx/2, step=dx)
    X, Y = np.meshgrid(steparr, steparr)
    return X, Y


def getXDomain():
    X, _ = getDomain()
    return np.transpose(X)


def getYDomain():
    _, Y = getDomain()
    return np.transpose(Y)


def getZeroMatrix():
    return np.zeros(MATSIZE)


def initialMatrix():
    return getYDomain()


class ResetSolutionButton(QtWidgets.QPushButton):
    """ button for resetting solution """
    def __init__(self, parent=None):
        super(ResetSolutionButton, self).__init__(parent)

        self.setText("Reset Solution")


class solutionView(pg.PlotWidget):
    """ main class for viewing 2D solution """
    def __init__(self, parent=None):
        super(solutionView, self).__init__(parent)

        self.viewMat = initialMatrix()

        self.img = pg.ImageItem(self.viewMat)
        self.kern = np.array([
            [0.0,  0.0, 0.25, 0.0, 0.0],
            [0.0,  0.25, 0.5, 0.25, 0.0],
            [0.25, 0.5,  1.0, 0.5,  0.25],
            [0.0,  0.25, 0.5, 0.25, 0.0],
            [0.0,  0.0, 0.25, 0.0, 0.0]])
        self.img.setDrawKernel(self.kern, mask=self.kern, center=(2, 2), mode='add')
        self.levels = [0, 1]
        self.img.setLevels(self.levels)

        self.temperature_lut = get_matplotlib_lut("CMRmap")
        self.pressure_lut = get_matplotlib_lut("nipy_spectral")
        self.density_lut = get_matplotlib_lut("viridis")
        self.img.setLookupTable(self.temperature_lut)       # initial colormap

        self.setTitle("Solution")
        x_axis = pg.AxisItem('bottom')
        y_axis = pg.AxisItem('left')
        axis_items = {'left': y_axis, 'bottom': x_axis}
        self.setAxisItems(axis_items)
        self.setLabel(axis='left', text='Y')
        self.setLabel(axis='bottom', text='X')
        self.showGrid(x=True, y=True, alpha=1)

        # will eventually remove these
        #self.hideAxis('bottom')
        #self.hideAxis('left')

        self.vb = self.getViewBox()
        self.vb.setBackgroundColor((100, 10, 34))
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.addItem(self.img)
        pen = pg.mkPen('y', width=3, style=QtCore.Qt.DashLine)
        self.vb.setBorder(pen)

    def resetSolution(self):
        self.viewMat = initialMatrix()
        self.img.setImage(self.viewMat)
        self.img.setLevels(self.levels)
        print("solution was reset!")

    def setPressureCmap(self):
        self.img.setLookupTable(self.pressure_lut)
        print("Pressure cmap appears!")

    def setTemperatureCmap(self):
        self.img.setLookupTable(self.temperature_lut)
        print("Temperature cmap appears!")

    def setDensityCmap(self):
        self.img.setLookupTable(self.density_lut)
        print("Density cmap appears!")

    def save_figure(self):
        exporter = pg.exporters.ImageExporter(self)
        exporter.export('testing!!!.png')

    #def paintEvent(self, ev):
    #    return super().paintEvent(ev)


class solutionChooser(QtWidgets.QWidget):
    """ main settings class for which solution to view """
    def __init__(self, parent=None):
        super(solutionChooser, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout()
        solution_label = QtGui.QLabel("Choose which solution to view")
        self.viewTemperatureRadio = QtGui.QRadioButton("Temperature")
        self.viewPressureRadio = QtGui.QRadioButton("Pressure")
        self.viewDensityRadio = QtGui.QRadioButton("Density")
        self.viewVelocityRadio = QtGui.QRadioButton("Velocity")

        self.viewTemperatureRadio.setChecked(True)

        layout.addWidget(solution_label)
        layout.addWidget(self.viewTemperatureRadio)
        layout.addWidget(self.viewPressureRadio)
        layout.addWidget(self.viewDensityRadio)
        layout.addWidget(self.viewVelocityRadio)

        self.viewTemperatureRadio.toggled.connect(lambda: self.radio_state(self.viewTemperatureRadio))
        self.viewPressureRadio.toggled.connect(lambda: self.radio_state(self.viewPressureRadio))
        self.viewDensityRadio.toggled.connect(lambda: self.radio_state(self.viewDensityRadio))
        self.viewVelocityRadio.toggled.connect(lambda: self.radio_state(self.viewVelocityRadio))

        self.setLayout(layout)

    def radio_state(self, b):
        if b.isChecked():
            print(b.text()+" is selected")
        else:
            print(b.text()+" is deselected")


class interactivityChooser(QtWidgets.QWidget):
    """ main settings class for which interactivity to have """
    def __init__(self, parent=None):
        super(interactivityChooser, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout()

        interactivity_label = QtGui.QLabel("Choose type of interactivity")
        self.wallSourceRadio = QtGui.QRadioButton("Wall")
        self.pressureSourceRadio = QtGui.QRadioButton("Pressure")
        self.temperatureSourceRadio = QtGui.QRadioButton("Temperature")

        self.wallSourceRadio.setChecked(True)

        layout.addWidget(interactivity_label)
        layout.addWidget(self.wallSourceRadio)
        layout.addWidget(self.pressureSourceRadio)
        layout.addWidget(self.temperatureSourceRadio)

        self.wallSourceRadio.toggled.connect(lambda: self.radio_state(self.wallSourceRadio))
        self.pressureSourceRadio.toggled.connect(lambda: self.radio_state(self.pressureSourceRadio))
        self.temperatureSourceRadio.toggled.connect(lambda: self.radio_state(self.temperatureSourceRadio))

        self.setLayout(layout)

    def radio_state(self, b):
        if b.isChecked():
            print(b.text()+" is selected")
        else:
            print(b.text()+" is deselected")


class Settings(QtWidgets.QWidget):
    """ main settings class """
    def __init__(self, parent=None):
        super(Settings, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout()

        self.reset_btn = ResetSolutionButton()
        self.sc = solutionChooser()
        self.ic = interactivityChooser()

        layout.addWidget(self.reset_btn)
        layout.addWidget(self.sc)
        layout.addWidget(self.ic)
        layout.addStretch(1)

        self.setLayout(layout)


class MainWindow(QtWidgets.QMainWindow):
    """ main class for InteractiveEuler """
    def __init__(self):
        super(MainWindow, self).__init__()

        # initialize UI
        self.init_ui()

        # setup euler solver

    def init_ui(self):
        #uiPath = os.path.join("interactiveeuler","ui","interactiveeuler.ui")
        #self.ui = uic.loadUi(uiPath)

        self.setWindowTitle('InteractiveEuler')
        self.resize(800, 600)

        bar = self.menuBar()
        # Creating menus using a QMenu object
        fileMenu = QtWidgets.QMenu("&File", self)
        bar.addMenu(fileMenu)
        # Creating menus using a title
        saveMenu = bar.addMenu("&Save")
        self.saveAction = QtWidgets.QAction("&Save", self)
        self.saveAction.triggered.connect(self.save_figure)
        helpMenu = bar.addMenu("&Help")


        main_layout = QtWidgets.QHBoxLayout()

        self.sv = solutionView()
        sl = Settings()
        sl.reset_btn.clicked.connect(self.resetSignal)

        # colormaps (and probably other things eventually)
        sl.sc.viewTemperatureRadio.toggled.connect(self.temperature_toggled)
        sl.sc.viewPressureRadio.toggled.connect(self.pressure_toggled)
        sl.sc.viewDensityRadio.toggled.connect(self.density_toggled)

        main_layout.addWidget(self.sv)
        main_layout.addWidget(sl)

        # status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.showMessage('Message in statusbar.')
        self.setStatusBar(self.statusBar)
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        s = f"(width, height) = ({width}, {height})"
        self.statusBar.showMessage(s)

        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def resetSignal(self):
        self.sv.resetSolution()

    def pressure_toggled(self):
        self.sv.setPressureCmap()

    def temperature_toggled(self):
        self.sv.setTemperatureCmap()

    def density_toggled(self):
        self.sv.setDensityCmap()

    def save_figure(self):
        self.sv.save_figure()
        print("called save figure!")


def main():
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('InteractiveEuler')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
