import pandas
import matplotlib.pyplot as plt

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
def drawRink():
    plt.clf()
    plt.xlim(0, ICE[0])
    plt.ylim(0, ICE[1])
    goalratio = GOAL_WIDTH / (2*ICE[1])
    plt.axvline(0.1, color='g', linewidth=2, ymin=0.5-goalratio, ymax=0.5+goalratio)
    plt.axvline(ICE[0]-0.1, color='g', linewidth=2, ymin=0.5-goalratio, ymax=0.5+goalratio)
    plt.savefig('test.png')

class Movement:
    def __init__(self, start, m, r, f, v):
        self.Mass = m
        self.Radius = r
        self.FrictionAcc = f*G
        self.Velocity = v
        self.Bounces = [start,]

    def velocity(self, t):
        return (self.Velocity[0] - t*self.FrictionAcc, self.Velocity[1] - t*self.FrictionAcc)

    def position(self, t):
        self.velocity(t)

    def move(self):
        print("nonsense")
        pass

def main(index):
    column = tuplify(DATA[index], NAMESIN, ["start", "velocity"])
    sim = Movement(column['start'], column['mass'], column['radius'], column['friction'], column['velocity'])
    sim.move()

if __name__=="__main__":
    for i in range(0,len(DATA.columns)):
        main(i)
