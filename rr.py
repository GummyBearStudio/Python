import numpy

A = numpy.array([[-2,1,0],
                [-1,0,1],
                [0,0,-1]])
J = numpy.array([[1,1,0],
                [0,1,1],
                [0,0,1]])

P = numpy.array([[2,3,3],
                [6,7,6],
                [8,4,2]])
Pv = numpy.linalg.inv(P)

print('P=')
print(P)
print('J=')
print(J)
print('P-1=')
print(Pv)

print('\n')

print(A)
print('=')
print(P@J@Pv)