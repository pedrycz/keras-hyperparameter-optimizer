import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-fraction', dest='data_fraction', default=0.46, type=float)
parser.add_argument('--test-fraction', dest='test_fraction', default=0.1, type=float)
parser.add_argument('--epochs', dest='epochs', default=1, type=int)
parser.add_argument('--early-stopping-patience', dest='early_stopping_patience', default=1, type=int)

args = parser.parse_args()
