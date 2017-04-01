# Python 2.7

"""
Lab 1 - An Application of Genetic Algorithms

Task:
Inscribe a triangle of the maximum area in a given ellipse.
Ellipse is defined as: (x/a)^2  + (y/b)^2 = 1
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

tstart = timer()

# Definition of parameters of the GA

N = 100     # number of units (chromosomes)
L = 36      # length of a chromosome (12 bits per vertex)
pc = 0.8    # crossover probability
pm = 0.001  # mutation probability
G = 0.8     # generation gap

# Parameters of the ellipse

a = 5
b = 3


#################
# The algorithm #
#################

# Maximum number with L bits
maxnum = 2**L - 1

# We'll use these a lot
a2 = float(a * a)
b2 = float(b * b)
Lthird = L//3
twoLthirds = 2 * Lthird
maxl3 = 2**Lthird - 1
piHalf = np.pi / 2
threePiHalf = 3. * piHalf
a2rec = 1. / a2

# Array of N chromosomes, each consisting of L bits
newgen = np.array([np.binary_repr(0, L)] * N)
oldgen = np.array([np.binary_repr(0, L)] * N)

# Vertices of the triangles; a vertex is defined by its angle to the positive x-axis in radians
V1 = np.empty(N)
V2 = np.empty(N)
V3 = np.empty(N)

# Coordinates of the vertices
x1 = np.empty(N)
y1 = np.empty(N)
x2 = np.empty(N)
y2 = np.empty(N)
x3 = np.empty(N)
y3 = np.empty(N)

# Fitness function
f = np.empty(N)

# Array that holds the maximum value of fitness function in every generation
Fmax = []

# The first generation
for i in range(N):
    tmp = np.random.random()   # a random number in [0.0, 1.0)
    newgen[i] = np.binary_repr(int(round(tmp * maxnum)), L)

# generation number counter
gencnt = 0

# condition for staying in the loop
cond = True

#The main loop
while cond:

    # Evaluation of the newly formed generation
    for i in range(N):
        V1[i] = (float(int(newgen[i][:Lthird], 2)) / maxl3) * 2.0 * np.pi
        V2[i] = (float(int(newgen[i][Lthird:twoLthirds], 2)) / maxl3) * 2.0 * np.pi
        V3[i] = (float(int(newgen[i][twoLthirds:], 2)) / maxl3) * 2.0 * np.pi

        # Coordinates of vertex V1
        if (V1[i] < piHalf) or (V1[i] > threePiHalf):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(V1[i])**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(V1[i])**2)/b2)))            
        y = x * math.tan(V1[i])
        x1[i] = x
        y1[i] = y

        # Coordinates of vertex V2
        if (V2[i] < piHalf) or (V2[i] > threePiHalf):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(V2[i])**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(V2[i])**2)/b2)))            
        y = x * math.tan(V2[i])
        x2[i] = x
        y2[i] = y

        # Coordinates of vertex V3
        if (V3[i] < piHalf) or (V3[i] > threePiHalf):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(V3[i])**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(V3[i])**2)/b2)))            
        y = x * math.tan(V3[i])
        x3[i] = x
        y3[i] = y

        # Lenghts of the triangle's edges
        la = math.sqrt((x2[i] - x1[i])**2 + (y2[i] - y1[i])**2)
        lb = math.sqrt((x3[i] - x1[i])**2 + (y3[i] - y1[i])**2)
        lc = math.sqrt((x3[i] - x2[i])**2 + (y3[i] - y2[i])**2)

        # Semiperimeter of the triangle
        s = (la + lb + lc) / 2.

        # Fitness function (Heron's formula)
        f[i] = math.sqrt(s * (s - la) * (s - lb) * (s - lc))

    # The highest (best) value of f
    maxf = np.amax(f)

    # Index of the highest value of f
    maxfindex = np.argmax(f)

    Fmax.append(maxf)

    # Plotting the result

    plt.figure("An Application of Genetic Algorithms")
    plt.hold(True)
    plt.title("Generation number {}\nThe best result: {:.4f}".format(gencnt + 1, maxf))
    plt.xlim(-a - 1, a + 1)
    plt.ylim(-b - 1, b + 1)

    # Drawing the ellipse
    ellipse = np.array([[0.] * 361, [0.] * 361], dtype = float)
    for i in range(361):
        theta = 2.*np.pi*i/360.
        if (theta <= piHalf) or (theta > threePiHalf):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(theta)**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(theta)**2)/b2)))
        y = x * math.tan(theta)
        ellipse[0][i] = x
        ellipse[1][i] = y    
    plt.plot(ellipse[0], ellipse[1], 'g', linewidth = 4.0)  # thick green line

    # Drawing the triangles that we got
    for i in range(N):
        if f[i] == maxf:
            # The best chromosome - the triangle with the largest area
            plt.plot([x1[i], x2[i], x3[i], x1[i]], [y1[i], y2[i], y3[i], y1[i]], 'r', linewidth = 4.0)  # thick red line
        else:
            # The other chromosomes (triangles); they are all inscribed in the ellipse, but they don't have the largest area
            plt.plot([x1[i], x2[i], x3[i], x1[i]], [y1[i], y2[i], y3[i], y1[i]], 'b', linewidth = 0.5)  # thin blue line
    
    # Hold the graph for a given amount of time in seconds
    plt.pause(0.1)
    plt.hold(False)
    plt.plot()
    
    ### Natural selection by the roulette wheel method ###

    oldgen = np.copy(newgen)

    # Cumulative function
    cumf = f.cumsum()

    # We let the best chromosome pass to the next generation directly.
    newgen[0] = oldgen[maxfindex]

    # We also let another randomly chosen (1-G)*N-1 chromosomes pass. Probability of their selection depends on f(i).
    for i in range(1, int(round((1-G)*N))):
        tmp = np.random.random() * cumf[-1]
        firstPositive, firstPositiveIndex = np.amax(np.sign(cumf - tmp)), np.argmax(np.sign(cumf - tmp))
        newgen[i] = oldgen[firstPositiveIndex]

    ### The rest of the new generation is formed by crossover (crossbreeding) ###

    # There are two parents, and two offsprings
    for i in range((N - int(round((1-G)*N)))//2):
        tmp = np.random.random() * cumf[-1]
        firstPositive, firstPositiveIndex = np.amax(np.sign(cumf - tmp)), np.argmax(np.sign(cumf - tmp))
        parent1 = oldgen[firstPositiveIndex]

        tmp = np.random.random() * cumf[-1]
        firstPositive, firstPositiveIndex = np.amax(np.sign(cumf - tmp)), np.argmax(np.sign(cumf - tmp))
        parent2 = oldgen[firstPositiveIndex]

        if np.random.random() < pc:
            # crossover
            crossPoint = int(round(np.random.random() * (L - 2))) + 1   # the crossover point can be after MSB and before LSB, thus L-2
            newgen[int(round((1-G)*N)) + 2*i] = parent1[:crossPoint] + parent2[crossPoint:]         # offspring 1
            newgen[int(round((1-G)*N)) + 2*i + 1] = parent2[:crossPoint] + parent1[crossPoint:]     # offspring 2
        else:
            # no crossover
            newgen[int(round((1-G)*N)) + 2*i] = parent1         # offspring 1
            newgen[int(round((1-G)*N)) + 2*i + 1] = parent2     # offspring 2

    ### Mutation ###

    for i in range(int(L * N * pm)):
        chromosomeIndex = int(round(np.random.random() * (N - 1)))
        bitPosition = int(round(np.random.random() * (L - 1)))
        newgen[chromosomeIndex] = newgen[chromosomeIndex][:bitPosition] + ('0' if newgen[chromosomeIndex][bitPosition] == '1' else '1') + newgen[chromosomeIndex][bitPosition + 1:]

    gencnt += 1
    
    # Exit condition - We want fitness functions of the first numchrom chromosomes to be inside of a given difference.
    numchrom = 10
    difference = 0.001
    f.sort()
    if abs(f[-1] - f[-numchrom]) < difference:
        cond = False
    
tend = timer()

print("The result is: {}".format(max(Fmax)))
print("The algorithm took {} generations, and {} seconds to complete.".format(gencnt, round(tend - tstart, 3)))
print("The maximum value of fitness function through generations:\n{}".format(Fmax))

plt.figure("The maximum value of fitness function through generations")
plt.title("The maximum value of fitness function through generations")
plt.xlim(0, gencnt - 1)
plt.plot(Fmax)
plt.show()
