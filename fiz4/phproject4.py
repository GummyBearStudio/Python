import pandas
import numpy
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

MARGIN = 1.0
STEP = 0.05
G = 6.7 * 1e-11

NAMESIN = []
NAMESOUT = []
for i in [1,2,3]:
    NAMESIN += [f"pos{i}", f"velocity{i}", f"mass{i}"]
    NAMESOUT += [f"pos{i}", f"velocity{i}"]
NAMESIN += ["t"]
NAMESOUT += ["first", "second"]
DATA = pandas.read_csv("input.txt", sep=';', header=None, names=NAMESIN).transpose()

""" Distance between points a and b """
def normDistance(a, b):
    return ((b[0] - a[0])**2.0 + (b[1] - a[1])**2.0)**0.5

""" Retrieves correct arctan angle """
def getVectorAngle(vec):
    Angle = 0.0
    if vec[0] != 0.0:
        Angle = math.atan(vec[1] / vec[0])
        # atan needs vector direction fix when Vx < 0
        if vec[0] < 0:
            Angle = math.pi + Angle
    else:
        Angle = math.pi / 2
        if vec[1] < 0:
            Angle = -Angle
    return Angle

class LineMovement:
    def __init__(self, start, v):
        self.Position = start
        self.setVelocity(v)

    def setVelocity(self, v):
        self.Velocity = v
        self.VNorm = (v[0]**2.0 + v[1]**2.0)**0.5
        self.Angle = getVectorAngle(self.Velocity)

    def getDistance(self, t):
        return self.VNorm * t

    def getPosition(self, t):
        distance = self.getDistance(t)
        dx = math.cos(self.Angle) * distance
        dy = math.sin(self.Angle) * distance
        return (self.Position[0] + dx, self.Position[1] + dy)

class Body:
    def __init__(self, start, mass, v):
        self.Movement = LineMovement(start, v)
        self.Mass = mass
        self.History = [start]

    def oneStep(self, force):
        self.Movement.Position = self.Movement.getPosition(STEP)
        (force / self.Mass) * STEP

    def gravity(self, other):
        angle = getVectorAngle(numpy.array(other.Movement.Position) - numpy.array(self.Movement.Position))
        strength = G * self.Mass * other.Mass / (normDistance(self.Movement.Position, other.Movement.Position)**2.0)
        return (strength * math.cos(angle), strength * math.sin(angle))

    def combineBody(self, other):
        mass = self.Mass + other.Mass
        avgpos = numpy.array(self.Movement.Position) * self.Mass + numpy.array(other.Movement.Position) * other.Mass
        avgv = numpy.array(self.Movement.Velocity) * self.Mass + numpy.array(other.Movement.Velocity) * other.Mass
        return Body(tuple(avgpos / mass), mass, tuple(avgv / mass))

def main(index, fhandle):
    print("-"*30, index+1, "-"*30)
    column = tuplify(DATA[index], NAMESIN, NAMESOUT)
    bodies = [ Body(column[f'pos{i}'], column[f'mass{i}'], column[f'velocity{i}']) for i in [1,2,3] ]
    # loop until all bodies merged
    while len(bodies) > 1:
        bodies = []
    # write files
    plt.clf()
    plt.savefig("{}.png".format(index+1))
    summary = [(0,0), [0,0], (0,0), (0,0)]
    fhandle.write('{0};{1};{2};{3};{4}\n'.format(*summary))

if __name__=="__main__":
    with open('output.txt', 'w') as f:
        for i in range(0,len(DATA.columns)):
            main(i, f)
