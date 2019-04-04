import pandas
import matplotlib.pyplot as plt
import math

# HELPERS

def getTuple(s):
    s = s.replace(' ', '')
    s = s.split(',')
    return (float(s[0][1:]), float(s[1][:-1]))

def tuplify(c, nms, tup):
    d = {n:c[n] for n in nms}
    for n in nms:
        if n in tup:
            d[n] = getTuple(d[n])
        else:
            d[n] = float(d[n])
    return d

# PHYSICS

ICE = (60.0, 40.0)
GOAL_WIDTH = 1.0
G = 10.0
NAMESIN = ["start", "mass", "radius", "friction", "velocity"]
NAMESOUT = ["stop", "time"]
DATA = pandas.read_csv("input.txt", sep=';', header=None, names=NAMESIN).transpose()

""" Ice Rink to file """
def drawRink(x, y, start=None):
    plt.clf()
    plt.xlim(0, ICE[0])
    plt.ylim(0, ICE[1])
    goalratio = GOAL_WIDTH / (2*ICE[1])
    plt.axvline(0.1, color='g', linewidth=2, ymin=0.5-goalratio, ymax=0.5+goalratio)
    plt.axvline(ICE[0]-0.1, color='g', linewidth=2, ymin=0.5-goalratio, ymax=0.5+goalratio)
    plt.plot(x, y)
    if start!=None:
        plt.plot(*start, 'go')

class LineMovement:
    def __init__(self, start, r, f, v):
        self.Radius = r
        self.Friction = f
        self.Velocity = v
        self.VNorm = (v[0]**2.0 + v[1]**2.0)**0.5
        self.Begin = start
        self.Angle = math.atan(self.Velocity[0] / self.Velocity[1])
        self.FVector = (f*G*math.cos(self.Angle), f*G*math.sin(self.Angle))

    def getStopTime(self):
        return self.VNorm / (self.Friction*G)

    def getDistance(self, t):
        lim = self.getStopTime()
        distance = lambda time: self.VNorm * time - 0.5*(time**2.0)*self.Friction*G
        return distance(t) if t < lim else distance(lim) # convert distance function to value at t

    def getWall(self):
        pass

    def getPosition(self, t):
        distance = self.getDistance(t)
        dx = math.cos(self.Angle) * distance
        dy = math.sin(self.Angle) * distance
        return (self.Begin[0] + dx, self.Begin[1] + dy)

def main(index):
    column = tuplify(DATA[index], NAMESIN, ["start", "velocity"])
    sim = LineMovement(column['start'], column['radius'], column['friction'], column['velocity'])
    axle = [x/100.0 for x in range(0,int(sim.getStopTime()*101))]
    drawRink([sim.getPosition(x)[0] for x in axle], [sim.getPosition(x)[1] for x in axle], column['start'])
    plt.savefig("{}.png".format(index+1))

if __name__=="__main__":
    for i in [0]:#range(0,len(DATA.columns)):
        main(i)
