import pandas
import matplotlib.pyplot as pyplt

# Tools for input processing
class Polynomial:
    def __init__(self, s):
        self.coef = []
        if type(s)==str:
            self.parse(s)
        elif type(s)==list:
            self.coef = s

    def calc(self, x):
        x = float(x)
        y = 0.0
        for i in range(0,len(self.coef)):
            y += self.coef[i] * (x**i)
        return y

    def parse(self, string):
        string = string.replace(' ','')
        x = {}
        while len(string) > 0:
            i = 1
            while i < len(string) and string[i] not in ['+','-']:
                i += 1
            if i<=0:
                break
            tmp = string[0:i]
            string = string[i:]
            i = tmp.find('x^')
            if i >= 0:
                x.update({int(tmp[i+2:]):float(tmp[:i])})
                continue
            i = tmp.find('x')
            if i >= 0:
                x.update({1:float(tmp[:i])})
            else:
                x.update({0:float(tmp)})
        flip = list(x.keys())
        self.coef = [0.0 for zero in range(0,1+max(flip))]
        for i in range(0,max(flip)+1):
            if i in flip:
                self.coef[i] = x[i]
            else:
                self.coef[i] = 0.0

def getTuple(s):
    s = s.replace(' ', '')
    s = s.split(',')
    return (float(s[0][1:]), float(s[1][:-1]))

def tuplify(c, nms, tup):
    d = {n:c[n] for n in nms}
    for n in tup:
        d[n] = getTuple(d[n])
    return d

# Physics code below
""" returns trajectory by time t """
def solveTrajectory(start, v, w):
    V = (v[0]+w[0], v[1]+w[1])
    return lambda t: ( start[0]+t*V[0], start[1]+t*V[1]-0.5*9.81*(t**2.0) )

""" returns either time when shell reaches peak or peak itself """
def solvePeak(v, w):
    return (v[1]+w[1])/9.81 # time when peak is reached, peak = trajectory(solvePeak)
    #return 0.5*((v[1]+w[1])**2.0)/9.81 # potential energy peak, ignores start point

""" velocity vector at time t """
def solveSpeed(t, v, w):
    V = (v[0]+w[0], v[1]+w[1])
    return [V[0], V[1]-9.81*t]

""" position where shell hit the ground """
def solvePosition(grd, tjct, left=0.1, right=100.0):
    x = (left+right)/2.0
    sh = tjct(x)
    if right-left <= 0.01:
        return sh, x
    dst = sh[1]-grd.calc(sh[0])
    if dst<0:
        return solvePosition(grd, tjct, left, x)
    elif dst==0.0:
        return sh, x
    else:
        return solvePosition(grd, tjct, x, right)

""" check whether shell hit the target """
def solveHit(pos, tg):
    margin = 0.05
    test = lambda i: (pos[i] > tg[i]-margin and pos[i] < tg[i]+margin)
    return test(0) and test(1)

INNAMES = ['start','target','velocity','wind','t1','t2','t3','w']
INTUPLES = INNAMES[:4]
INPUT = pandas.read_csv('input.txt', sep=';', header=None, names=INNAMES).transpose()
OUTNAMES = ['position','maxh','v1','v2','v3','hit']
OUTPUT = pandas.DataFrame(index=OUTNAMES)

for i in range(0,len(INPUT.columns)):
    column = tuplify(INPUT[i], INNAMES, INTUPLES)
    ground = Polynomial(column['w'])
    trajectory = solveTrajectory(column['start'], column['velocity'], column['wind'])
    maxh = trajectory(solvePeak(column['velocity'], column['wind']))[1]
    #maxh = solvePeak(column['velocity'], column['wind']) + column['start'][1]
    velocities = [ solveSpeed(float(column['t1']), column['velocity'], column['wind']), solveSpeed(float(column['t2']), column['velocity'], column['wind']), solveSpeed(float(column['t3']), column['velocity'], column['wind']) ]
    position, hittime = solvePosition(ground, trajectory)
    hit = solveHit(position, column['target'])
    # generate plots
    displacement = abs(column['start'][0] - column['target'][0])
    edge = min(column['start'][0], column['target'][0])
    axisx = [edge+displacement*a/100.0 for a in range(-50,151)]
    pyplt.clf()
    pyplt.plot(axisx, [ground.calc(a) for a in axisx])
    axisx = [hittime*a/100.0 for a in range(0,101)]
    pyplt.plot([trajectory(a)[0] for a in axisx], [trajectory(a)[1] for a in axisx])
    pyplt.plot(*column['start'], 'go')
    pyplt.plot(*column['target'], 'ro')
    pyplt.savefig('map{0}.png'.format(i+1))
    # round values to 2 decimal places
    position = ( round(position[0],2), round(position[1],2) )
    for k in range(0,len(velocities)):
        velocities[k] = [ round(velocities[k][0],2), round(velocities[k][1],2) ]
    # add to dataframe
    OUTPUT[i] = [ position, round(maxh,2), velocities[0], velocities[1], velocities[2], hit ]
print(INPUT,'\n')
print(OUTPUT)
OUTPUT = OUTPUT.transpose()
with open('output.txt', 'w') as f:
    OUTPUT.to_csv(f, sep=';', header=False, index=False)
