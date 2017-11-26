import time
from deap import creator, base, tools, algorithms

from function import optimize, attrs

start = time.time()

# evolutionary tools
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# evolutionary parameters
population_size = 10
generations = 5
crossover_probability = 0.1
mating_probability = 0.3
mutation_probability = 0.1
tournament_size = 2
toolbox = base.Toolbox()
toolbox.register("individual", tools.initCycle, creator.Individual, attrs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", optimize)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=crossover_probability)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)

# create population and run algorithm
population = toolbox.population(n=population_size)

for gen in range(generations):
    print
    print("GENRATION " + str(gen))
    offspring = algorithms.varAnd(population, toolbox, cxpb=mating_probability, mutpb=mutation_probability)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# get best population representative
top1 = tools.selBest(population, k=1)[0]

# print best representative
print
print("Best result for parameters: " + str(top1))
end = time.time()
print(end - start)