import pandas
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

ICE = (60.0, 40.0)
GOAL_WIDTH = 1.0
G = 10.0
NAMESIN = ["start", "mass", "radius", "friction", "velocity"]
NAMESOUT = ["stop", "time"]
DATA = pandas.read_csv("fiz2\\input.txt", sep=';', header=None, names=NAMESIN).transpose()

""" Ice Rink to file """
def drawRink(start=None):
    plt.clf()
    plt.xlim(0, ICE[0])
    plt.ylim(0, ICE[1])
    goalratio = GOAL_WIDTH / (2*ICE[1])
    plt.axvline(0.1, color='g', linewidth=2, ymin=0.5-goalratio, ymax=0.5+goalratio)
    plt.axvline(ICE[0]-0.1, color='g', linewidth=2, ymin=0.5-goalratio, ymax=0.5+goalratio)
    if start!=None:
        plt.plot(*start, 'go')

def checkGoal(pos):
    upper = (ICE[1]+GOAL_WIDTH) / 2
    lower = (ICE[1]-GOAL_WIDTH) / 2
    return (pos[0] <= 0 or pos[0] >= ICE[0]) and pos[1] < upper and pos[1] > lower

class LineMovement:
    def __init__(self, start, v, r, f):
        self.Radius = r
        self.Friction = f
        self.Begin = start
        self.setVelocity(v)

    def setVelocity(self, v):
        self.Velocity = v
        self.VNorm = (v[0]**2.0 + v[1]**2.0)**0.5
        if self.Velocity[0] != 0.0:
            self.Angle = math.atan(self.Velocity[1] / self.Velocity[0])
        else:
            self.Angle = math.pi / 2
        self.FVector = (self.Friction*G*math.cos(self.Angle), self.Friction*G*math.sin(self.Angle))

    def update(self, t, b):
        self.Begin = self.getPosition(t)
        self.setVelocity(self.getVelocity(t))
        if b == 'l':
            self.Velocity = (abs(self.Velocity[0]), self.Velocity[1])
        elif b == 'r':
            self.Velocity = (-abs(self.Velocity[0]), self.Velocity[1])
        elif b == 't':
            self.Velocity = (self.Velocity[0], -abs(self.Velocity[1]))
        elif b == 'b':
            self.Velocity = (self.Velocity[0], abs(self.Velocity[1]))

    def borders(self):
        return {'l':self.Radius, 'r':ICE[0]-self.Radius, 'b':self.Radius, 't':ICE[1]-self.Radius}

    def getHitTime(self):
        def inrange(t):
            pos = self.getPosition(t)
            return pos[0] >= 0 and pos[1] >= 0 and pos[0] <= ICE[0] and pos[1] <= ICE[1]
        borders = self.borders()
        x = sympy.Symbol('x', real=True, positive=True)
        result = self.getStopTime()
        bouncewall = None
        for wall in borders:
            axle = 0 if wall in ['l', 'r'] else 1
            solutions = sympy.solve(self.getPosition(x)[axle] - borders[wall], x)
            if len(solutions) > 0:
                m = min(solutions)
                if m < result and inrange(float(m)):
                    result = float(m)
                    bouncewall = wall
        self.update(result, bouncewall)
        return result

    def getVelocity(self, t):
        return (self.Velocity[0] - t*self.FVector[0], self.Velocity[1] - t*self.FVector[1])

    def getStopTime(self):
        return self.VNorm / (self.Friction*G)

    def getDistance(self, t):
        return self.VNorm * t - 0.5*(t**2.0)*self.Friction*G

    def getPosition(self, t):
        distance = self.getDistance(t)
        dx = math.cos(self.Angle) * distance
        dy = math.sin(self.Angle) * distance
        return (self.Begin[0] + dx, self.Begin[1] + dy)

def main(index, fhandle):
    print("-"*30, index+1, "-"*30)
    column = tuplify(DATA[index], NAMESIN, ["start", "velocity"])
    drawRink(column['start'])
    sim = LineMovement(column['start'], column['velocity'], column['radius'], column['friction'])
    # params
    finish = sim.getStopTime()
    bounces = [column['start']]
    elapsed = 0.0
    inside = True
    # loop every bounce
    print("Slide duration: {} s".format(finish))
    while elapsed < finish and inside:
        traveltime = sim.getHitTime()
        elapsed += traveltime
        bounces.append(sim.Begin)
        if traveltime==0:
            break
        else:
            inside = not checkGoal(bounces[-1])
    df = pandas.DataFrame(bounces)
    print(df)
    plt.plot(df[0], df[1])
    plt.savefig("{}.png".format(index+1))
    if inside:
        fhandle.write('({0:.2f}, {1:.2f}); {2:.2f}; {3}\n'.format(*bounces[-1], finish, '; '.join(["({0:.2f}, {0:.2f})".format(*bounces[x]) for x in range(1, len(bounces)-1)])))
    else:
        fhandle.write('(out); {0:.2f}; {1}\n'.format(finish, '; '.join(["({0:.2f}, {0:.2f})".format(*bounces[x]) for x in range(1, len(bounces)-1)])))

if __name__=="__main__":
    with open('output.txt', 'w') as f:
        for i in range(0,len(DATA.columns)):
            main(i, f)
