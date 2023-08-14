import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from datetime import datetime

import NonContinuousFunctions
import visualize_fun_trajec_move
from functions import *
import time
from NonContinuousFunctions import *
#from NonContinuousRotatedHybridCompositeFunc import *
from datetime import date
###############limits of variables for each function#####################


DIMENSIONS = 2 #without y axis
#define hyper-parameters
START_RESULTANT = -.82# start step size
END_RESULTANT = .00001   # end step size
START_ANGLE =30
END_ANGLE = 60
NUM_OF_STEPS = 200
NUM_OF_STEPS =15
fitness = rastrigin2
REFINEMENT_PREC = 0.0000001
low = -2*(np.pi)
up =2*(np.pi)
outside_trajectoroies_count = 0

# print([[random.randint(1,10) for _ in range(49)] for _ in range(5)])
#############################################
def iterative_refinement(isUnderneath, y_func, x_space, y_space, xsteps, yStep, cur_round):
    #jump in the middle of the step until crossing point is detected
    while abs(y_space-y_func)>REFINEMENT_PREC and yStep!=0:
        for j in range(0, DIMENSIONS):
            xsteps[j] = xsteps[j]/2
        yStep = yStep/2

        if (isUnderneath==False and y_space>y_func) or (isUnderneath==True and y_space<y_func):
            for j in range(0, DIMENSIONS):
                x_space[j] += xsteps[j]
            y_space += yStep
        else:
            for j in range(0, DIMENSIONS):
                x_space[j] -= xsteps[j]
            y_space -= yStep
         
        y_func = fitness(x_space)
        #if step gets too small, exit because we have a satisfactorily accurate solution
        if abs(yStep)<REFINEMENT_PREC:
            y_space = y_func
            
    return x_space, y_func
#############################################
def BPround(xvalues, y, i,DIMENSIONS, low, up, fitness,NUM_OF_STEPS, START_RESULTANT , END_RESULTANT ,START_ANGLE, END_ANGLE, MAX_ITER):
    #define cooling schedules for resultant and angle
    resultant = START_RESULTANT - i*(START_RESULTANT-END_RESULTANT)/MAX_ITER
    a = START_ANGLE - i*(START_ANGLE-END_ANGLE)/MAX_ITER

    #setting direction randomly #find resultant
    xsteps=[]
    numerator=0
    resultant_step = 0
    for j in range(DIMENSIONS):
        xsteps.append(random.uniform(-1, 1))
        numerator += -(xsteps[j]**2)*(math.sin(math.radians(a))**2)
        resultant_step+=xsteps[j]**2
        
    yStep = -math.sqrt(numerator/((math.sin(math.radians(a))**2) - 1))
    resultant_step = math.sqrt(resultant_step + yStep**2)
    #find factor
    f=resultant/resultant_step
    
    #calculate steps of the vector for the given direction #start current vector
    x_space = []
    for j in range(0, DIMENSIONS):
        xsteps[j] = xsteps[j]*abs(f)
        x_space.append(xvalues[j] + xsteps[j])
    yStep = yStep*abs(f)
    y_space = y + yStep
    
    y_func = fitness(x_space)
    
    if y_space < y_func:
        isUnderneath = True
    else:
        isUnderneath = False
        
    for countSteps in range(NUM_OF_STEPS):#steps begin here
        if min(x_space)<low or max(x_space)>up:
            global outside_trajectoroies_count
            outside_trajectoroies_count += 1
            break
        
        if (isUnderneath==False and y_space < y_func) or (isUnderneath==True and y_space > y_func) or y_space==y_func:####crossing detected implementation
            xvalues, y = iterative_refinement(isUnderneath, y_func, x_space, y_space, xsteps, yStep, i)
            break
        
        for j in range(0, DIMENSIONS):
            x_space[j] += xsteps[j]
        y_space += yStep
        y_func = fitness(x_space)
        
    return xvalues, y
#############################################
# for START_RESULTANT in sr:
#     for END_ANGLE in ea:
times=[]
results=[]
exp=0
start_time1=time.process_time()

plot_xvals = []
plot_xvals_ys = []

def main(DIMENSIONS, low, up, fitness,NUM_OF_STEPS, START_RESULTANT , END_RESULTANT ,START_ANGLE, END_ANGLE, MAX_ITER):
    xvals=[]
    xvals = [random.uniform(low, up) for i in range(DIMENSIONS)]
    y = fitness(xvals)
    print("start ", y)
    for i in range(MAX_ITER): # number of rounds
        xvals, y = BPround(xvals, y, i, DIMENSIONS, low, up, fitness,NUM_OF_STEPS, START_RESULTANT , END_RESULTANT ,START_ANGLE, END_ANGLE, MAX_ITER)

    results.append(y) #collect accuracy and time results of each algorithm run

    print("global xvals ",xvals, " || global y ", y)
    return  xvals