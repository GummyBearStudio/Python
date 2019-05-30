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

    def _getImpulse(self, other):
        normal = numpy.array(self.Movement.Position) - numpy.array(other.Movement.Position)
        dst = (normal @ normal)**0.5
        normal = normal / dst
        dv = numpy.array(self.Movement.Velocity) - numpy.array(other.Movement.Velocity)
        strength = (dv @ normal) * 2 * self.Mass * other.Mass / (self.Mass + other.Mass)
        return normal * strength

    def transferEnergy(self, other, bht):
        if bht == None:
            return
        impulse = self._getImpulse(other)
        energy = self.Movement.VNorm**2 * self.Mass + other.Movement.VNorm**2 * other.Mass
        self.Movement.setVelocity(tuple(numpy.array(self.Movement.Velocity) - impulse / self.Mass))
        other.Movement.setVelocity(tuple(numpy.array(other.Movement.Velocity) + impulse / other.Mass))
        print("Kinetic energy debug:",energy,'=',self.Movement.VNorm**2 * self.Mass + other.Movement.VNorm**2 * other.Mass)

def main(index, fhandle):
    print("-"*30, index+1, "-"*30)
    column = tuplify(DATA[index], NAMESIN, NAMESOUT)
    print(column)
    return
    bodies = [Body(column['start1'], column['mass1'], column['velocity1'])]
    # loop until all bodies merged
    while len(bodies) > 1:
        bodies = []
    # write files
    plt.clf()
    plt.savefig("{}.png".format(index+1))
    summary = [(0,0), (0,0), 0, 0, 0]
    fhandle.write('{0};{1};{2};{3};{4}\n'.format(*summary))

if __name__=="__main__":
    with open('output.txt', 'w') as f:
        for i in range(0,len(DATA.columns)):
            main(i, f)
