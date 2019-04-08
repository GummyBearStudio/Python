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

MARGIN = 0.005
ICE = (60.0, 40.0)
GOAL_WIDTH = 1.0
G = 10.0
NAMESIN = ["start", "mass", "radius", "friction", "velocity"]
NAMESOUT = ["stop", "time"]
DATA = pandas.read_csv("input.txt", sep=';', header=None, names=NAMESIN).transpose()

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
    goalmargin = 0.1
    upper = ((ICE[1]+GOAL_WIDTH) / 2) + goalmargin
    lower = ((ICE[1]-GOAL_WIDTH) / 2) - goalmargin
    return (pos[0] <= goalmargin or pos[0] >= ICE[0]-goalmargin) and pos[1] < upper and pos[1] > lower

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
            if self.Velocity[0] < 0: # atan needs vector direction fix when Vx < 0
                self.Angle = math.pi + self.Angle
        else:
            self.Angle = math.pi / 2
            if self.Velocity[1] < 0:
                self.Angle = -self.Angle
        self.FVector = (-self.Friction*G*math.cos(self.Angle), -self.Friction*G*math.sin(self.Angle))

    def update(self, t, b):
        self.Begin = self.getPosition(t)
        new_v = self.getVelocity(t)
        if b == 'l':
            new_v = (abs(new_v[0]), new_v[1])
        elif b == 'r':
            new_v = (-abs(new_v[0]), new_v[1])
        elif b == 't':
            new_v = (new_v[0], -abs(new_v[1]))
        elif b == 'b':
            new_v = (new_v[0], abs(new_v[1]))
        self.setVelocity(new_v)

    def borders(self):
        return {'l':self.Radius, 'r':ICE[0]-self.Radius, 'b':self.Radius, 't':ICE[1]-self.Radius}

    def getHitTime(self):
        borders = self.borders()
        x = sympy.Symbol('x', real=True, positive=True)
        result = self.getStopTime()
        bouncewall = None
        for wall in borders:
            axle = 0 if wall in ['l', 'r'] else 1
            solutions = sympy.solve(self.getPosition(x+10e-14)[axle] - borders[wall], x)
            if len(solutions) > 0:
                m = float(min(solutions))
                if m < result:
                    pos = self.getPosition(m)
                    if pos[0] >= -MARGIN and pos[1] >= -MARGIN and pos[0] <= ICE[0]+MARGIN and pos[1] <= ICE[1]+MARGIN:
                        result = m
                        bouncewall = wall
        self.update(result, bouncewall)
        return result

    def getVelocity(self, t):
        return (self.Velocity[0] + t*self.FVector[0], self.Velocity[1] + t*self.FVector[1])

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
    times = [0.0]
    elapsed = 0.0
    inside = True
    # loop every bounce
    print("Slide duration: {} s".format(finish))
    while elapsed < finish and inside:
        times.append(sim.getHitTime())
        elapsed += times[-1]
        bounces.append(sim.Begin)
        if times[-1]==0:
            break
        else:
            inside = not checkGoal(bounces[-1])
    df = pandas.DataFrame(data={'x':[round(b[0], 2) for b in bounces], 'y':[round(b[1], 2) for b in bounces], 'delta t':[round(t, 3) for t in times]})
    print(df)
    plt.plot(df['x'], df['y'])
    plt.savefig("{}.png".format(index+1))
    if inside:
        fhandle.write('({0:.2f}, {1:.2f}); {2:.2f}; {3}\n'.format(*bounces[-1], finish, '; '.join(["({0:.2f}, {0:.2f})".format(*bounces[x]) for x in range(1, len(bounces)-1)])))
    else:
        fhandle.write('(out); {0:.2f}; {1}\n'.format(finish, '; '.join(["({0:.2f}, {0:.2f})".format(*bounces[x]) for x in range(1, len(bounces)-1)])))

if __name__=="__main__":
    with open('output.txt', 'w') as f:
        for i in range(0,len(DATA.columns)):
            main(i, f)
