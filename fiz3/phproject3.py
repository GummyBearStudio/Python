import pandas
import numpy
import matplotlib.pyplot as plt
import math
import sympy

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

MARGIN = 0.001
BALL_MASS = 0.017
BALL_R = 0.03
FRICTION = 0.015
ENERGY_RATIO = 1.0 - 0.3
TABLE = (2.7, 1.35)
HOLE_R = 0.1 / 2.0
G = 9.81
NAMESIN = ["white", "velocity", "colored"]
NAMESOUT = ["white", "colored", "ballhits", "whitewallhits", "colorwallhits"]
DATA = pandas.read_csv("input.txt", sep=';', header=None, names=NAMESIN).transpose()

""" Table to file """
def draw(markers=None):
    mscale = 200
    plt.clf()
    plt.xlim(0, TABLE[0])
    plt.ylim(0, TABLE[1])
    x = [0, TABLE[0]/2.0, TABLE[0]]
    plt.plot(x, [0, 0, 0], 'ro', markersize=mscale*HOLE_R)
    plt.plot(x, [TABLE[1], TABLE[1], TABLE[1]], 'ro', markersize=mscale*HOLE_R)
    if markers!=None and len(markers)==2:
        plt.plot(*markers[0], 'bo', markersize=mscale*BALL_R)
        plt.plot(*markers[1], 'go', markersize=mscale*BALL_R)

""" Check whether ball fell into a hole """
def checkGoal(pos):
    x = [0, TABLE[0]/2.0, TABLE[0]]
    error = MARGIN + HOLE_R
    if (pos[1] > -error and pos[1] < error) or (pos[1] > TABLE[1]-error and pos[1] < TABLE[1]+error):
        for i in x:
            if pos[0] > i - error and pos[0] < i + error:
                return True
    return False

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

def getPointDistance(angle, cross, point):
    a = math.tan(angle)
    b = cross[1] - a*cross[0]
    return (a*point[0] + b - point[1]) / ((1 + a**2.0)**0.5)

class LineMovement:
    def __init__(self, start, v):
        self.borders = {'l':BALL_R, 'r':TABLE[0]-BALL_R, 'b':BALL_R, 't':TABLE[1]-BALL_R}
        self.Begin = start
        self.setVelocity(v)

    def setVelocity(self, v):
        self.Velocity = v
        self.VNorm = (v[0]**2.0 + v[1]**2.0)**0.5
        self.Angle = getVectorAngle(self.Velocity)
        self.FVector = (-FRICTION*G*math.cos(self.Angle), -FRICTION*G*math.sin(self.Angle))

    def update(self, t, b):
        self.Begin = self.getPosition(t)
        new_v = self.getVelocity(t)
        if b != None:
            energy_loss = ENERGY_RATIO ** 0.5
            new_v = (new_v[0]*energy_loss, new_v[1]*energy_loss)
            if b == 'l':
                new_v = (abs(new_v[0]), new_v[1])
            elif b == 'r':
                new_v = (-abs(new_v[0]), new_v[1])
            elif b == 't':
                new_v = (new_v[0], -abs(new_v[1]))
            elif b == 'b':
                new_v = (new_v[0], abs(new_v[1]))
        self.setVelocity(new_v)

    def getHitTime(self):
        x = sympy.Symbol('x', real=True, positive=True)
        result = self.getStopTime()
        bouncewall = None
        for wall in self.borders:
            axle = 0 if wall in ['l', 'r'] else 1
            solutions = sympy.solve(self.getPosition(x+10e-14)[axle] - self.borders[wall], x)
            if len(solutions) > 0:
                m = float(min(solutions))
                if m < result:
                    pos = self.getPosition(m)
                    if pos[0] >= -MARGIN and pos[1] >= -MARGIN and pos[0] <= TABLE[0]+MARGIN and pos[1] <= TABLE[1]+MARGIN:
                        result = m
                        bouncewall = wall
        self.update(result, bouncewall)
        return result

    def getVelocity(self, t):
        if self.VNorm == 0:
            t = 0
        return (self.Velocity[0] + t*self.FVector[0], self.Velocity[1] + t*self.FVector[1])

    def getStopTime(self):
        return self.VNorm / (FRICTION*G)

    def getDistance(self, t):
        return self.VNorm * t - 0.5*(t**2.0)*FRICTION*G

    def getPosition(self, t):
        distance = self.getDistance(t)
        dx = math.cos(self.Angle) * distance
        dy = math.sin(self.Angle) * distance
        return (self.Begin[0] + dx, self.Begin[1] + dy)

class Ball:
    def __init__(self, pos, v=(0, 0), mass=BALL_MASS):
        self.Movement = LineMovement(pos, v)
        self.Mass = mass
        self.bounces = [pos]
        self.times = [0.0]
        self.elapsed = 0.0
        self.inside = True

    def addBounce(self, p, t):
        self.bounces.append(p)
        self.times.append(t)
        self.elapsed += t

    def getBallHit(self, other, left=0.01, right=None):
        if right == None:
            right = self.Movement.getStopTime()
        time = (right+left) / 2.0
        dst = normDistance(self.Movement.getPosition(time), other.Movement.getPosition(time))
        if dst - 2*BALL_R <= MARGIN:
            return time
        if right-left < MARGIN:
            return None
        h = 0.001
        dstprim = normDistance(self.Movement.getPosition(time+h), other.Movement.getPosition(time+h)) - dst
        if dstprim < 0:
            return self.getBallHit(other, time, right)
        else:
            return self.getBallHit(other, left, time)

    def _getImpulse(self, other):
        normal = numpy.array(self.Movement.Begin) - numpy.array(other.Movement.Begin)
        normal = normal / ((normal @ normal)**0.5)
        dmom = numpy.array(self.Movement.Velocity) * self.Mass - numpy.array(other.Movement.Velocity) * other.Mass
        strength = dmom @ normal
        return normal * strength

    def transferEnergy(self, other, bht):
        if bht == None:
            return
        self.Movement.update(bht, None)
        other.Movement.update(bht, None)
        self.addBounce(self.Movement.Begin, bht)
        other.addBounce(other.Movement.Begin, bht)
        impulse = self._getImpulse(other)
        energy = self.Movement.VNorm**2 * self.Mass + other.Movement.VNorm**2 * other.Mass
        self.Movement.setVelocity(tuple(numpy.array(self.Movement.Velocity) - impulse / self.Mass))
        other.Movement.setVelocity(tuple(numpy.array(other.Movement.Velocity) + impulse / other.Mass))

    def loop(self, counter, cousins=[]):
        for c in cousins:
            ballt = self.getBallHit(c)
            if ballt!=None:
                counter += 1
                self.transferEnergy(c, ballt)
        wallt = self.Movement.getHitTime()
        self.addBounce(self.Movement.Begin, wallt)
        self.inside = not checkGoal(self.Movement.Begin)
        return counter

    def dataSheet(self):
        return pandas.DataFrame(data={'x':[round(b[0], 2) for b in self.bounces], 'y':[round(b[1], 2) for b in self.bounces], 'delta t':[round(t, 3) for t in self.times]})

def main(index, fhandle):
    print("-"*30, index+1, "-"*30)
    column = tuplify(DATA[index], NAMESIN, NAMESIN)
    draw([column['white'], column['colored']])
    balls = [Ball(column['white'], column['velocity']), Ball(column['colored'])]
    counter = 0
    # loop every bounce
    while balls[0].inside and balls[1].inside:
        counter = balls[0].loop(counter, [balls[1],])
        counter = balls[1].loop(counter, [balls[0],])
        if balls[0].times[-1]>-10e-9 and balls[0].times[-1]<10e-9 and balls[1].times[-1]>-10e-9 and balls[1].times[-1]<10e-9:
            break
    white = balls[0].dataSheet()
    color = balls[1].dataSheet()
    print(white)
    print(color)
    print('Ball hits: ', counter)
    # write files
    plt.plot(white['x'], white['y'])
    plt.plot(color['x'], color['y'])
    plt.savefig("{}.png".format(index+1))
    # TODO: write output.txt
    end = '(faul)'
    cend = '(score)'
    fhandle.write(';\n')

if __name__=="__main__":
    with open('output.txt', 'w') as f:
        for i in range(0,len(DATA.columns)):
            main(i, f)
