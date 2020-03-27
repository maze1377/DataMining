import math
import random
import sys

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QBrush
from PyQt5.QtWidgets import QApplication, QMainWindow


def calculateCovariance(X):
    meanX = np.mean(X, axis=0)
    lenX = X.shape[0]
    X = X - meanX
    covariance = X.T.dot(X) / lenX
    return covariance


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]


# params
MAX = 2000
'''
this param for set Ellipse position :)
'''
# Ellipse Param
F1 = 300
F2 = 300
A1 = 200  # X1 ->
A2 = 200  # X2 |
RR = 0  # counterclockwise rotation

data = []
while True:
    x = random.random() * 690
    y = random.random() * 690
    if pow((x - F1) / A1, 2) + pow((y - F2) / A2, 2) <= 1:
        data.append(rotate([F1, F2], [x, y], RR))
    if len(data) == MAX:
        break

# cal
print("Covarinace")

x = np.array(data)

print("Shape of array data:\n", np.shape(x))

# print("data:\n", x)
CovMatrix = calculateCovariance(x)
print("covariance matrix of x:\n", CovMatrix)

w, v = np.linalg.eig(CovMatrix)
print("eigenvalues  matrix of x:\n", w)
print("right eigenvectors of a square array matrix of x:\n", v)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Data Mining"
        self.top = 100
        self.left = 100
        self.width = 700
        self.height = 700

        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def paintEvent(self, e):
        painter = QPainter(self)
        # painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.setBrush(QBrush(Qt.green, Qt.DiagCrossPattern))
        painter.drawRect(0, 0, 700, 700)

        painter.setPen(QPen(Qt.black, 8, Qt.SolidLine))
        painter.setFont(QtGui.QFont('Decorative', 16))
        painter.drawLine(2, 2, 2, 700)
        painter.drawText(10, 650, "x2")
        painter.drawLine(2, 2, 700, 2)
        painter.drawText(650, 25, "x1")

        # painter.rotate(RR)
        # painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        # painter.drawEllipse(F1 - A1, F2 - A2, 2 * A1, 2 * A2)
        # painter.rotate(-RR)

        painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
        for x, y in data:
            painter.drawPoint(int(x), int(y))

        # draw line of distributes:
        painter.setPen(QPen(Qt.blue, 6, Qt.SolidLine))
        # distrubteX2 = w[1] / (w[1] + w[0])
        distrubteX2 = 1
        endpointX2 = rotate([F1, F2], [F1, (A2 + F2) * distrubteX2], RR)
        painter.drawLine(F1, F2, int(endpointX2[0]), int(endpointX2[1]))
        # distrubteX1 = w[0] / (w[1] + w[0])
        distrubteX1 = 1
        endpointX1 = rotate([F1, F2], [(F1 + A1) * distrubteX1, F2], RR)
        painter.drawLine(F1, F2, int(endpointX1[0]), int(endpointX1[1]))


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
