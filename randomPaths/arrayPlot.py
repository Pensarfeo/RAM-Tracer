import matplotlib.pyplot as plt



# print(pix)
# plt.imshow(pixels, cmap='gray',interpolation='nearest')
# print('---------------')
# plt.plot()


# import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import bezier
import math
# sphinx_gallery_thumbnail_number = 2


def gaussian(x, y, x0, y0, sigma):
    coef = 1 #sqrt(2*math.pi*sigma)
    coef = 1/coef
    xExp = (x-x0)**2
    yExp = (y-y0)**2
    expVal = -(xExp + yExp)/(2*(sigma**2))
    return math.exp(expVal)

def nicePrintArr(arr):
    for arre in arr:
        print(arre)

def genRandomPoint(i):
    return [random.random(), random.random()]

def genBezierPath(n = 10, nB = 25):
    toBezerPoints = [genRandomPoint(i) for i in range(0, n)]
    toBezerPoints = list(zip(*toBezerPoints))
    points = np.asfortranarray(toBezerPoints)
    curve = bezier.Curve(points, degree=2)

    xBzPoints = []
    yBzPoints = []

    for i in range(0, nB + 1):
        xBzPoint, yBzPoint = np.hstack(curve.evaluate(i/nB))
        xBzPoints.append(xBzPoint)
        yBzPoints.append(yBzPoint)
    
    return curve, toBezerPoints, [xBzPoints, yBzPoints]


sigma = 5

bCurve, [xToBPoints, yToBPoints], bPoints = genBezierPath(n = 15)

# for bPoint in bPoints:
import pdb
def mapGaussian(x0, y0, picSize = None, arr = None):

    _picSize = picSize if (picSize and (not arr)) else len(arr) 
    _arr = arr if arr else [[0] * picSize] * _picSize
    x0Scaled = x0*_picSize
    y0Scaled = y0*_picSize

    for xi in range(0, picSize):
        _arr[xi] = list(map(
            lambda y: _arr[xi][y[0]] + gaussian(xi, y[0], x0Scaled, y0Scaled, sigma),
            enumerate([0]*_picSize)
        ))
    return _arr


arr = mapGaussian(0.5, 0.5, picSize = 100)
# pdb.set_trace()

# nicePrintArr(arr)

# fig, ax = plt.subplots()
plt.imshow(arr, cmap='gray')

plt.show()