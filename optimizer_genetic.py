import time
from deap import creator, base, tools, algorithms

from function import optimize, attrs
from arguments import args

start = time.time()

# evolutionary tools
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# evolutionary parameters
toolbox = base.Toolbox()
toolbox.register("individual", tools.initCycle, creator.Individual, attrs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", optimize)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=args.crossover_probability)
toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)

# create population and run algorithm
population = toolbox.population(n=args.population_size)

for gen in range(args.generations):
    print
    print("GENRATION " + str(gen))
    offspring = algorithms.varAnd(population, toolbox, cxpb=args.manting_probability, mutpb=args.mutation_probability)
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
