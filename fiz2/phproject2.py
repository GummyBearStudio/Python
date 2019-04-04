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
DATA = pandas.read_csv("d:\\Projects\\Python\\fiz2\\input.txt", sep=';', header=None, names=NAMESIN).transpose()

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

def bouncev(v, place):
    dst = {'l':place[0], 'r':ICE[0]-place[0], 't':ICE[1]-place[1], 'b':place[1]}
    closest = 'l'
    for k in dst.keys():
        if dst[k] < dst[closest]:
            closest = k
    if closest == 'l':
        v = (abs(v[0]), v[1])
    elif closest == 'r':
        v = (-abs(v[0]), v[1])
    elif closest == 't':
        v = (v[0], -abs(v[1]))
    elif closest == 'b':
        v = (v[0], abs(v[1]))
    return v

class LineMovement:
    def __init__(self, start, v, r, f):
        self.Radius = r
        self.Friction = f
        self.Velocity = v
        self.VNorm = (v[0]**2.0 + v[1]**2.0)**0.5
        self.Begin = start
        self.Angle = math.atan(self.Velocity[0] / self.Velocity[1])
        self.FVector = (f*G*math.cos(self.Angle), f*G*math.sin(self.Angle))

    def borders(self):
        return {'l':self.Radius, 'r':ICE[0]-self.Radius, 'b':self.Radius, 't':ICE[1]-self.Radius}

    def getHitTime(self):
        borders = self.borders()
        x = sympy.Symbol('x', real=True, positive=True)
        hits = sympy.solve(self.getPosition(x)[1] - borders['t'], x)
        hits += sympy.solve(self.getPosition(x)[1] - borders['b'], x)
        hits += sympy.solve(self.getPosition(x)[0] - borders['l'], x)
        hits += sympy.solve(self.getPosition(x)[0] - borders['r'], x)
        ts = []
        stop = self.getStopTime()
        for i in hits:
            if i <= stop:
                ts.append(i)
        print(ts)
        return min(ts) if len(ts) > 0 else 0

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

def main(index):
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
    while elapsed < finish and inside:
        traveltime = sim.getHitTime()
        elapsed += traveltime
        bounces.append(sim.getPosition(traveltime))
        if traveltime == 0:
            inside = False
            updatedv = sim.getVelocity(0.0)
        elif elapsed >= finish:
            updatedv = (0, 0)
            del bounces[-1]
            bounces.append(sim.getPosition(finish-elapsed+traveltime))
            break
        else:
            updatedv = bouncev(sim.getVelocity(traveltime), bounces[-1])
        sim = LineMovement(bounces[-1], updatedv, column['radius'], column['friction'])
    df = pandas.DataFrame(bounces)
    print("Slide duration: {} s".format(finish))
    print(df)
    plt.plot(df[0], df[1])
    plt.savefig("{}.png".format(index+1))
    # output: (end pos); finish time; bounces...

if __name__=="__main__":
    for i in range(0,len(DATA.columns)):
        main(i)
