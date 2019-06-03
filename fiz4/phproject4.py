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
STEP = 0.5 #0.05
G = 6.7 * 1e-11
N = 3 # extra param

NAMESIN = []
NAMESOUT = []
for i in range(1, N+1):
    NAMESIN += [f"pos{i}", f"velocity{i}", f"mass{i}"]
    NAMESOUT += [f"pos{i}", f"velocity{i}"]
NAMESIN += ["t"]
NAMESOUT += [f"{i}c" for i in range(1, N)]
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

    def getPosition(self, t, a=0):
        distance = self.getDistance(t) + 0.5*a*(t**2)
        dx = math.cos(self.Angle) * distance
        dy = math.sin(self.Angle) * distance
        return (self.Position[0] + dx, self.Position[1] + dy)

class Body:
    def __init__(self, start, mass, v):
        self.Movement = LineMovement(start, v)
        self.Mass = mass
        self.History = []

    def __str__(self):
        return "Body: "+str(self.Mass)+" at "+str(self.Movement.Position)

    def updateStep(self, force):
        acc = force / self.Mass
        self.History.append(self.Movement.Position)
        self.Movement.Position = self.Movement.getPosition(STEP, float(acc@acc))
        self.Movement.setVelocity( numpy.array(self.Movement.Velocity) + acc * STEP )

    def sumGravity(self, others):
        f = numpy.zeros(2)
        for x in others:
            f += self.gravity(x)
        return f

    def gravity(self, other):
        angle = getVectorAngle(numpy.array(other.Movement.Position) - numpy.array(self.Movement.Position))
        strength = G * self.Mass * other.Mass / (normDistance(self.Movement.Position, other.Movement.Position)**2.0)
        return numpy.array([strength * math.cos(angle), strength * math.sin(angle)])

    def combineBody(self, other):
        mass = self.Mass + other.Mass
        avgpos = numpy.array(self.Movement.Position) * self.Mass + numpy.array(other.Movement.Position) * other.Mass
        avgv = numpy.array(self.Movement.Velocity) * self.Mass + numpy.array(other.Movement.Velocity) * other.Mass
        combined = Body(tuple(avgpos / mass), mass, tuple(avgv / mass))
        return combined

def main(index, fhandle):
    print("-"*30, index+1, "-"*30)
    column = tuplify(DATA[index], NAMESIN, NAMESOUT)
    bodies = [ Body(column[f'pos{i}'], column[f'mass{i}'], column[f'velocity{i}']) for i in range(1, N+1) ]
    print([str(b) for b in bodies])
    print("Sim duration: {0}".format(column['t']))
    stories = []
    collisions = []
    elapsed = 0.0
    # loop until all bodies merged
    while len(bodies) > 1 and elapsed < column['t']:
        elapsed += STEP
        for x in range(0, len(bodies)):
            bodies[x].updateStep(bodies[x].sumGravity(bodies[:x] + bodies[x+1:]))
        for x in bodies:
            for y in bodies:
                if x != y and normDistance(y.Movement.Position, x.Movement.Position) < MARGIN:
                    bodies.append(x.combineBody(y))
                    collisions.append(bodies[-1].Movement.Position)
                    stories.append((x.History, x.Mass))
                    bodies.remove(x)
                    stories.append((y.History, y.Mass))
                    bodies.remove(y)
    print([str(b) for b in bodies])
    # write files
    plt.clf()
    for c in collisions:
        plt.plot(*c, 'ro')
    for stry in stories:
        plt.plot([s[0] for s in stry[0]], [s[1] for s in stry[0]], label=stry[1])
    plt.legend()
    plt.savefig("{}.png".format(index+1))
    summary = [(0,0), [0,0]]*N + [(0,0),]*(N-1)
    for i in range(0, len(collisions)):
        summary[len(summary)-N+1+i] = (round(collisions[i][0], 2), round(collisions[i][1], 2))
    print(summary)
    fhandle.write(str(summary[0]))
    for i in range(1, len(summary)):
        fhandle.write('; {0}'.format(summary[i]))
    fhandle.write('\n')

if __name__=="__main__":
    with open('output.txt', 'w') as f:
        for i in range(0,len(DATA.columns)):
            main(i, f)
