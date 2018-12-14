# import matplotlib.path as mpath
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import pdb
# import random
# Path = mpath.Path

# fig, ax = plt.subplots()

# def genRandomPoint(i):
#     return (random.random(), random.random() )

# def genRandomPath(n = 5):
#     points = [genRandomPoint(i) for i in range(0, n)]

#     pathType = [Path.MOVETO]

#     for i in range(1, n-1):
#         pathType.append(Path.CURVE4)
#     pathType.append(Path.CURVE4)
#     pp1 = mpatches.PathPatch(
#         Path(points, pathType),
#         fc="none",
#         transform=ax.transData
#     )
#     return points, pp1

# points, path = genRandomPath(n = 10)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.add_patch(path)
# ax.set_title('The red point should be on the path')
# firstPoint = points[0]
# lastPoint = points[-1]
# points = points[1:-1]
# xPoints, yPoints = list(zip(*points))
# ax.plot([firstPoint[0]], [firstPoint[1]], "g^")
# ax.plot([lastPoint[0]], [lastPoint[1]], "bs")
# ax.plot(xPoints, yPoints, "ro")
# plt.show()

import numpy as np
import bezier
import matplotlib.pyplot as plt
# import seaborn
import random
import pdb
 
# seaborn.set()

plt.show()

def randSign():
    return random.getrandbits(1) * 2 - 1
    
def shiftPoint(point, min, max):
    return point - ((max - min)/2) + ((max + min)/2)

def rand(min, max):
    randR = random.random() * (max - min)
    return shiftPoint(randR, min, max)

class Bezier:
    def __init__(self, n = 10, points = [], pertub = False):
        self.pointApprox = {}
        self.n = n        
        self.points = self.genPoints(n, points)
        self.points = self.perturbInitPoints(points) if pertub else self.points
        self.curve = bezier.Curve(self.points, degree=10)        

    def perturbInitPoints(self, points):
        maxP = 0.11
        tempPoints = list(zip(*points))
        perturbPoint = lambda x: (x[0] + rand(-maxP, maxP), x[1] + rand(-maxP, maxP)) 
        perturbPoint = np.array(list(map(perturbPoint, tempPoints))).transpose()
        perturbPoints = np.asfortranarray(perturbPoint)       

        return perturbPoints

    def genPoints(self, n, points = None):
        if len(points) > 1:
            return points
        
        points = [self.genRandomPoint(i) for i in range(0, n)]  
        points = list(zip(*points))
        return np.asfortranarray(points)

    def genRandomPoint(self, i):
        return [random.random(), random.random()]

    def plotPoints(self, n = 25):
        s_vals = np.linspace(0.0, 1.0, n)
        self.pointApprox[n] = self.curve.evaluate_multi(s_vals)
        return self.pointApprox[n]

    def centerPoints(self, xs, ys):
        shiftBPoint = lambda x, mx: x - (((mx[1] + mx[0])*0.5) - 0.5)

        xsMinMax = [min(xs), max(xs)]
        xs = list(map(lambda x: shiftBPoint(x, xsMinMax), xs))

        ysMinMax = [min(ys), max(ys)]
        ys = list(map(lambda x: shiftBPoint(x, ysMinMax), ys))

        return xs, ys
    
    def stretchPoints():
        return

ppoints = 20
figSizePx = 28
ticks = False
startEndPoints = False
inBetweenPoints = False
myDpi = 96
nBezierPoints = 4
annotate = False

for j in range(0, 25):
    bPath = Bezier(n = nBezierPoints)
    [xToBPoints, yToBPoints] = bPath.points
    
    bPoints = bPath.plotPoints(n = 25)

    [xBPoints, yBPoints] = bPath.centerPoints(*bPoints)
    


    # pdb.set_trace()
    # ax = bCurve.plot(num_pts = ppoints)
    # (num_pts=256)


    fig, ax = plt.subplots(figsize=(figSizePx/myDpi, figSizePx/myDpi), dpi=myDpi)

    if not ticks:
        plt.axis('off')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.autoscale_view('tight')

    # ax.set_facecolor((1.0, 0.47, 0.42))
    ax.set_aspect(1)
    ax.set_xlim(-0, 1)
    ax.set_ylim(-0, 1)
    # Points
    ax.plot(xBPoints, yBPoints, "-", linewidth=figSizePx/10)

    if annotate:
        for points in zip(xToBPoints, yToBPoints):
            print(points[0:1], points[-1:-2:-1], points)
            ax.plot([points[0]], [points[1]], marker='^', c='r', ms=2)
        for i,j,n in zip(xToBPoints, yToBPoints, list(range(0, nBezierPoints))):
            ax.annotate(str(n), xy=(i,j))
    
    if startEndPoints:
        ax.plot([xBPoints[0]], [yBPoints[0]], "r^")
        ax.plot([xBPoints[-1]], [yBPoints[-1]], "bs")
    
    if inBetweenPoints: 
        ax.plot(xBPoints[1:-1], yBPoints[1:-1], "ro", ms=1)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    print(data.shape)
    # print(fig.canvas.get_width_height(), fig.canvas.get_width_height()[::-1], fig.canvas.get_width_height()[::-1] + (3,))
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,) )
    # data = data[0] + data[1] + data[1]
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=myDpi)
    plt.imshow(data)
    # print(data[1])

    # for n in range(0, 5):
    #     perPath = Bezier(points = bPath.points, pertub = True)
    #     ax.plot(*perPath.plotPoints(n = 555), "-")



    # ax.plot([0,0,1,1], [0,1,0,1], c='orange')
    # ax.plot(xToBPoints[1:-1], yToBPoints[1:-1], "ro")
    # plt.savefig(
    # 'testplot'+str(j)+'.png',
    #     bbox_inches='tight',
    #     transparent="True",
    #     facecolor='white',
    #     pad_inches=0,
    #     dpi=32
    # )
    plt.show()
    # plt.savefig()
    
