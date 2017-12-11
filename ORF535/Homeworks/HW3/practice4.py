import numpy as np
import numpy
import itertools

'''rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
print(rranges)'''
'''print(slice(-4, 4, 0.25))
print(slice(1,5,2))'''
'''x = [X * 0.5 for X in range(0, 3)]
y = [X * 0.5 for X in range(0, 3)]
z = [X * 0.5 for X in range(0, 3)]

[X, Y, Z] = np.meshgrid(x, y, z)
print(X)
print(Y)
print(Z)'''
'''xaxis = np.linspace(0.0, 1.0, num=100, endpoint=True)
print(xaxis)'''
'''x = [X * 0.01 for X in range(0, 101)]
print(x)'''

def is_sum_one(tup):
    sum = 0
    for i in range(len(tup)):
        sum += tup[i]
    if sum == 1:
        return True
    else:
        return False

x = np.linspace(0.0, 1.0, num=2, endpoint = False)
y = np.linspace(0.0, 1.0, num=100, endpoint = False)
z = np.linspace(0.0, 1.0, num=2, endpoint = False)
w1 = list()
w2 = list()
w3 = list()
print(y)
for i in range(y.size):
    w1.append(float(round(y[i], 2)))
    w2.append(float(round(y[i], 2)))
    w3.append(float(round(y[i], 2)))
w1.append(1)
w2.append(1)
w3.append(1)
print(w1)
'''comb = itertools.combinations(w1, 1)
#comb_list = list()
comb_list = list(comb)
#print(list(comb))
print(comb_list)'''

'''number = [53, 64, 68, 71, 77, 82, 85]


results = itertools.combinations(number,4)
# convert the combination iterator into a numpy array
col_one = numpy.array(list(results))
print(col_one)

# calculate average of col_one
col_one_average = numpy.mean(col_one, axis = 1).astype(int)

# I don't actually create col_two, as I never figured out a good way to do it
# But since I only need the sum, I figure that out by subtraction
col_two_average = (numpy.sum(number) - numpy.sum(col_one, axis = 1)) / 3

dif = col_one_average - col_two_average

print(dif)'''
'''a = [[1,2,3],[4,5,6],[7,8,9,10]]
c = list(itertools.product(*a))
print(c)'''
w = list()
w.append(w1)
w.append(w2)
w.append(w3)
print(w)
d = list(itertools.product(*w))
#print(d)

sum_to_one = list()
for i in range(len(d)):
    if(is_sum_one(d[i]) is True):
        sum_to_one.append(d[i])
    else:
        continue

print(sum_to_one)
print(len(sum_to_one))



#print(y.size)