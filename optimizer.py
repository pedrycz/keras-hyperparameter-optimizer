import random
from deap import creator, base, tools, algorithms

# minimalized function
def optimize(params):
    return [-abs(200 - params[0]) + -abs(500 - params[1]) + -abs(800 - params[2])]

# function parameters
toolbox = base.Toolbox()
toolbox.register("attr_1", random.randint, 0, 1000)
toolbox.register("attr_2", random.randint, 0, 1000)
toolbox.register("attr_3", random.randint, 0, 1000)
attrs = [toolbox.attr_1, toolbox.attr_2, toolbox.attr_3]

# evolutionary tools
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# evolutionary parameters
population_size = 100
generations = 300
crossover_probability = 0.05
mating_probability = 0.5
mutation_probability = 0.1
tournament_size = 3
toolbox.register("individual", tools.initCycle, creator.Individual, attrs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", optimize)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=crossover_probability)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)

# create population and run algorithm
population = toolbox.population(n=population_size)
for gen in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=mating_probability, mutpb=mutation_probability)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# get best population representative
top1 = tools.selBest(population, k=1)[0]

# print best representative
print(top1)

