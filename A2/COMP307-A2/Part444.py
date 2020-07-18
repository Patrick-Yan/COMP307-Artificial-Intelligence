import pandas as pd
import operator
import math
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

data = pd.read_csv("/Users/Patrick/Desktop/satellite.csv",header=0)


features = data.iloc[:, :-1]
labels = data.iloc[:, -1].values

for x in range(len(labels)):
    if labels[x] == "'Normal'":
        labels[x] = 1
    else:  # =="'Anomaly'"
        labels[x] = 0

# print(len(features))
# print(len(labels))
#
# print(features[1][1])
# print(labels)

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)  # +
pset.addPrimitive(operator.sub, 2)  # -
pset.addPrimitive(operator.mul, 2)  # *
pset.addPrimitive(protectedDiv, 2)  # /

pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

pset.renameArguments(ARG0="V1",ARG1="V2",ARG2="V3",ARG3="V4",ARG4="V5",ARG5="V6",ARG6="V7",ARG7="V8",ARG8="V9",ARG9="V10",ARG10="V11"
                     ,ARG11="V12",ARG12="V13",ARG13="V14",ARG14="V15",ARG15="V16",ARG16="V17",ARG17="V18",ARG18="V19",ARG19="V20",ARG20="V21",ARG21="V22"
                     ,ARG22="V23",ARG23="V24",ARG24="V25",ARG25="V26",ARG26="V27",ARG27="V28",ARG28="V29",ARG29="V30",ARG30="V31",ARG31="V32",
                     ARG32="V33",ARG33="V34",ARG34="V35",ARG35="V36")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalCla(individual, points):
    func = toolbox.compile(expr=individual)

    correct = 0
    for x in range(len(points)):
        predict = func(points.iloc[x][0])

        actual = labels[x]
        if predict < 0:
            predict = 0
        else:
            predict = 1

        if predict == actual:
            correct = correct + 1

    return correct / len(labels),  # Accuracy


toolbox.register("evaluate", evalCla, points=features)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

if __name__ == "__main__":

    random.seed(30)
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.1, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print(log)
    # print(hof[1])
    # print(math.fsum(diff) / len(allX))

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, Best results are %s" % (best_ind, best_ind.fitness.values))
    # return pop, log, hof
