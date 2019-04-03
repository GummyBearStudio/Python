import pandas
import matplotlib.pyplot as plt

# HELPERS

def getTuple(s):
    s = s.replace(' ', '')
    s = s.split(',')
    return (float(s[0][1:]), float(s[1][:-1]))

def tuplify(c, nms, tup):
    d = {n:c[n] for n in nms}
    for n in tup:
        d[n] = getTuple(d[n])
    return d

# PHYSICS

ICE = (60.0, 40.0)
GOAL_WIDTH = 1.0
G = 10.0
NAMESIN = ["start", "mass", "radius", "friction", "velocity"]
NAMESOUT = ["stop", "time"]
DATA = pandas.read_csv("input.txt", sep=';', header=None, names=NAMESIN).transpose()

def main(index):
    column = tuplify(DATA[index], NAMESIN, ["start", "velocity"])
    print(column)

if __name__=="__main__":
    main(0)
