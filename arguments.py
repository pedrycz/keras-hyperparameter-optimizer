import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-fraction', dest='data_fraction', default=0.46, type=float)
parser.add_argument('--test-fraction', dest='test_fraction', default=0.1, type=float)
parser.add_argument('--epochs', dest='epochs', default=25, type=int)
parser.add_argument('--early-stopping-patience', dest='early_stopping_patience', default=5, type=int)
parser.add_argument('--population-size', dest='population_size', default=10, type=int)
parser.add_argument('--generations', dest='generations', default=5, type=int)
parser.add_argument('--crossover-probability', dest='crossover_probability', default=0.1, type=float)
parser.add_argument('--manting-probability', dest='manting_probability', default=0.3, type=float)
parser.add_argument('--mutation-probability', dest='mutation_probability', default=0.1, type=float)
parser.add_argument('--tournament-size', dest='tournament_size', default=2, type=int)

args = parser.parse_args()
