import random

from deap import creator, base, tools, algorithms
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential

from data import get_training_and_test_data

x, y = get_training_and_test_data()


# minimized function
def optimize(params):
    model = Sequential()
    model.add(LSTM(units=params[0] * 50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(params[1] * 50))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    return model.fit(x, y, epochs=2, batch_size=32, validation_split=0.15).history.get('val_acc')[-1]


# function parameters
toolbox = base.Toolbox()
toolbox.register("attr_1", random.randint, 1, 2)
toolbox.register("attr_2", random.randint, 1, 2)
attrs = [toolbox.attr_1, toolbox.attr_2]

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
