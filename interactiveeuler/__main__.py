"""
This is the module InteractiveEuler.

It is used to create a fun, interactive fluids solver.
"""

import sys

import numpy as np
from matplotlib import cm

import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets

import fluid as fl


N = 100
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


def quiver(X, Y, U, V):
    """Generates vector field visualization on grid (X,Y) -- each of
    which are numpy matrices of the X and Y coordinates that one wants
    to plot at. (U,V) are numpy matrices that describe the vector field
    at the supplied points (X,Y).

    Note
    ----
    A normalization to the vector field is not supplied in the current version
    of the code.

    Return values are expected to be used by
    setData(self.datax, self.datay, connect='pairs')

    """

    # step 1. position vectors on grid
    x0 = np.ndarray.flatten(X)
    y0 = np.ndarray.flatten(Y)
    #print(x0)
    #print(len(x0))

    #print(x0.shape)
    #print(np.ndarray.flatten(U).shape)
    # step 2. compute end points
    x1 = x0 + np.ndarray.flatten(U)
    #print(x1)
    y1 = y0 + np.ndarray.flatten(V)

    # step 3 compute scaling
    mult_scale = 1.0

    # step 4. interspace two arrays
    xdata = np.ravel([x0, x1], order='F')
    ydata = np.ravel([y0, y1], order='F')

    #for count in range(len(xdata)):
    #    print(count, xdata[count], ydata[count])

    # step 5. apply scaling
    xdata *= mult_scale
    ydata *= mult_scale

    # step 6. return data
    return xdata, ydata


class ResetSolutionButton(QtWidgets.QPushButton):
    """ button for resetting solution """
    def __init__(self, parent=None):
        super(ResetSolutionButton, self).__init__(parent)

        self.setText("Reset Solution")


class solutionView(pg.PlotWidget):
    """ main class for viewing 2D solution """
    def __init__(self, parent=None):
        super(solutionView, self).__init__(parent)

        self.ti = 0
        self.nVelocitySkip = np.maximum(10, round(N / 15))
        print(self.nVelocitySkip)
        self.viewMat = initialMatrix()

        self.img = pg.ImageItem(self.viewMat)
        self.kern = np.array([
            [0.0,  0.0, 0.25, 0.0, 0.0],
            [0.0,  0.25, 0.5, 0.25, 0.0],
            [0.25, 0.5,  1.0, 0.5,  0.25],
            [0.0,  0.25, 0.5, 0.25, 0.0],
            [0.0,  0.0, 0.25, 0.0, 0.0]])
        self.img.setDrawKernel(self.kern, mask=self.kern,
                               center=(2, 2), mode='add')
        self.levels = [0, 1]
        self.img.setLevels(self.levels)

        self.pressure_lut = get_matplotlib_lut("CMRmap")
        self.scalar_lut = get_matplotlib_lut("nipy_spectral")
        self.img.setLookupTable(self.pressure_lut)       # initial colormap

        self.setTitle("Solution")
        x_axis = pg.AxisItem('bottom')
        y_axis = pg.AxisItem('left')
        axis_items = {'left': y_axis, 'bottom': x_axis}
        self.setAxisItems(axis_items)
        self.setLabel(axis='left', text='Y')
        self.setLabel(axis='bottom', text='X')
        self.showGrid(x=True, y=True, alpha=1)

        # will eventually remove these
        # self.hideAxis('bottom')
        # self.hideAxis('left')

        self.vb = self.getViewBox()
        self.vb.setBackgroundColor((100, 10, 34))
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.addItem(self.img)

        # quiver field for velocity
        self.grid = fl.Grid(N)
        self.p1 = self.plot()
        self.setLimits(xMin=0, xMax=N, yMin=0, yMax=N)
        self.plot_flowfield()

        pen = pg.mkPen('y', width=3, style=QtCore.Qt.DashLine)
        self.vb.setBorder(pen)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def resetSolution(self):
        self.viewMat = initialMatrix()
        self.img.setImage(self.viewMat)
        self.img.setLevels(self.levels)
        print("solution was reset!")

    def setPressureCmap(self):
        self.img.setLookupTable(self.pressure_lut)
        print("Pressure cmap appears!")

    def setScalarCmap(self):
        self.img.setLookupTable(self.scalar_lut)
        print("Scalar cmap appears!")

    def save_figure(self):
        exporter = pg.exporters.ImageExporter(self)
        exporter.export('testing!!!.png')

    def plot_flowfield(self):
        # possibly could use pg.arrayToQPath(x, y, connect='pairs')
        self.U = 10*0.5*self.grid.X
        self.V = 10*0.5*self.grid.Y
        tempDataX, tempDataY = quiver((N-1)*self.grid.X[0::self.nVelocitySkip, 0::self.nVelocitySkip],
                                      (N-1)*self.grid.Y[0::self.nVelocitySkip, 0::self.nVelocitySkip],
                                      self.U[0::self.nVelocitySkip, 0::self.nVelocitySkip],
                                      self.V[0::self.nVelocitySkip, 0::self.nVelocitySkip])
        self.p1.setData(tempDataX, tempDataY, connect='pairs')

    def update_plot(self):

        # velocity field
        self.U += np.random.normal(size=(N, N))
        self.V += np.random.normal(size=(N, N))
        tempDataX, tempDataY = quiver((N-1)*self.grid.X[0::self.nVelocitySkip, 0::self.nVelocitySkip],
                                      (N-1)*self.grid.Y[0::self.nVelocitySkip, 0::self.nVelocitySkip],
                                      self.U[0::self.nVelocitySkip, 0::self.nVelocitySkip],
                                      self.V[0::self.nVelocitySkip, 0::self.nVelocitySkip])
        self.p1.setData(tempDataX, tempDataY, connect='pairs')
        self.ti += 1

    def toggle_quiver(self, command):
        if command == 'remove':
            self.removeItem(self.p1)
            print("Removing Flow Field")
        elif command == 'add':
            self.addItem(self.p1)
            print("Adding Flow Field")


class solutionChooser(QtWidgets.QWidget):
    """ main settings class for which solution to view """
    def __init__(self, parent=None):
        super(solutionChooser, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout()
        solution_label = QtGui.QLabel("Choose which solution to view")
        self.viewPressure = QtGui.QRadioButton("Pressure")
        self.viewScalar = QtGui.QRadioButton("Scalar Field")
        self.viewVelocity = QtGui.QCheckBox("Velocity")

        self.viewPressure.setChecked(True)
        self.viewVelocity.setChecked(True)

        layout.addWidget(solution_label)
        layout.addWidget(self.viewPressure)
        layout.addWidget(self.viewScalar)
        layout.addWidget(self.viewVelocity)

        self.setLayout(layout)


class interactivityChooser(QtWidgets.QWidget):
    """ main settings class for which interactivity to have """
    def __init__(self, parent=None):
        super(interactivityChooser, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout()

        interactivity_label = QtGui.QLabel("Choose type of interactivity")
        self.wallSource = QtGui.QRadioButton("Wall +")
        self.wallSink = QtGui.QRadioButton("Wall -")
        self.pressureSource = QtGui.QRadioButton("Pressure +")
        self.pressureSink = QtGui.QRadioButton("Pressure -")
        self.scalarSource = QtGui.QRadioButton("Scalar +")
        self.scalarSink = QtGui.QRadioButton("Scalar -")
        self.VelocitySource = QtGui.QRadioButton("Velocity +")

        self.wallSource.setChecked(True)

        layout.addWidget(interactivity_label)
        layout.addWidget(self.wallSource)
        layout.addWidget(self.wallSink)
        layout.addWidget(self.pressureSource)
        layout.addWidget(self.pressureSink)
        layout.addWidget(self.scalarSource)
        layout.addWidget(self.scalarSink)
        layout.addWidget(self.VelocitySource)

        self.setLayout(layout)


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
        sl.sc.viewScalar.toggled.connect(self.scalar_toggled)
        sl.sc.viewPressure.toggled.connect(self.pressure_toggled)
        sl.sc.viewVelocity.toggled.connect(lambda: self.velocity_toggled(sl.sc.viewVelocity))

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

    def scalar_toggled(self):
        self.sv.setScalarCmap()

    def velocity_toggled(self, btn):
        if btn.isChecked():
            self.sv.toggle_quiver("add")
        else:
            self.sv.toggle_quiver("remove")

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
