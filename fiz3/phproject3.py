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
BALL_MASS = 0.017
BALL_R = 0.03
FRICTION = 0.015
ENERGY_RATIO = 1.0 - 0.3
TABLE = (2.7, 1.35)
HOLE_R = 0.1 / 2.0
G = 9.81
NAMESIN = ["white", "velocity", "colored"]
NAMESOUT = ["white", "colored", "bounces", "whitewallhits", "colorwallhits"]
DATA = pandas.read_csv("input.txt", sep=';', header=None, names=NAMESIN).transpose()

""" Table to file """
def draw(markers=None):
    plt.clf()
    plt.xlim(0, TABLE[0])
    plt.ylim(0, TABLE[1])
    if markers!=None:
        plt.plot(*markers, 'go')

""" Check whether ball well into a hole """
def checkGoal(pos):
    goalmargin = 0.1
    pass

class LineMovement:
    def __init__(self, start, v):
        self.borders = {'l':BALL_R, 'r':TABLE[0]-BALL_R, 'b':BALL_R, 't':TABLE[1]-BALL_R}
        self.Begin = start
        self.setVelocity(v)

    def setVelocity(self, v):
        self.Velocity = v
        self.VNorm = (v[0]**2.0 + v[1]**2.0)**0.5
        if self.Velocity[0] != 0.0:
            self.Angle = math.atan(self.Velocity[1] / self.Velocity[0])
            # atan needs vector direction fix when Vx < 0
            if self.Velocity[0] < 0:
                self.Angle = math.pi + self.Angle
        else:
            self.Angle = math.pi / 2
            if self.Velocity[1] < 0:
                self.Angle = -self.Angle
        self.FVector = (-FRICTION*G*math.cos(self.Angle), -FRICTION*G*math.sin(self.Angle))

    def update(self, t, b):
        self.Begin = self.getPosition(t)
        new_v = self.getVelocity(t)
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

def main(index, fhandle):
    print("-"*30, index+1, "-"*30)
    column = tuplify(DATA[index], NAMESIN, NAMESIN)
    draw(column['start'])
    sim = LineMovement(column['start'], column['velocity'])
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
