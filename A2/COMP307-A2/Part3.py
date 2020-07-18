import operator
import math
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# load data
allX = []
allY = []
# # # # # # # # # # # # # # Please change path here# # # # # # # # # # # # #
f = open("/Users/Patrick/Desktop/COMP307/A2/ass2_data/part3/regression", "r")
with f as filehandle:
    for line in filehandle:
        if line.startswith("X"):
            continue

        line_array = line.split()
        allX.append(float(line_array[0]))
        allY.append(float(line_array[-1]))
# print(allX)
# print(allY)
f.close()

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
# pset.addPrimitive(operator.)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0="x")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)



def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    diff = []
    for x in range(len(allX)):
        diff.append( (func(points[x]) - allY[x])**2 )
    return (math.fsum(diff) / len(points)),  # MSE


toolbox.register("evaluate", evalSymbReg, points=allX)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))


def main():
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
    return pop, log, hof


if __name__ == "__main__":
    main()
