import copy
import random

import numpy as np
from keras.utils import np_utils

import arguments

args = arguments.args

shortest_sequence_length = 109
longest_sequence_length = 205
longest_sequence_length_with_trimmed_zeros = 182
number_of_character_classes = 20  # 'a''b''c''d''e''g''h''l''m''n''o''p''q''r''s''u''v''w''y''z'
zero_point_line_that_can_be_skipped = "0,0,0"
single_sequence_end_line = ",,"
padding_vector = [0.0, 0.0, 0.0]


def get_data():
    x = []
    y = []

    with open('data/classes.txt') as f:
        character_classes = f.readlines()
    for character_class in character_classes[0].split('|'):
        y.append(int(character_class) - 1)

    y = np_utils.to_categorical(y, number_of_character_classes)

    with open('data/sequences.csv') as f:
        x_y_pressure_points = f.readlines()

    single_sequence = []

    for point in x_y_pressure_points:
        if zero_point_line_that_can_be_skipped in point:
            continue

        if single_sequence_end_line in point:
            for i in range(longest_sequence_length_with_trimmed_zeros - len(single_sequence)):
                single_sequence.insert(0, padding_vector)

            x.append(copy.deepcopy(single_sequence))

            single_sequence = []
            continue

        single_sequence.append([])
        for point_element in point.split(','):
            single_sequence[-1].append(float(point_element))

    x_y = list(zip(x, y))
    random.shuffle(x_y)
    x, y = zip(*x_y)

    data_count = int(args.data_fraction * len(x))

    x = x[:data_count]
    y = y[:data_count]

    test_count = int(args.test_fraction * len(x))

    x_train = np.array(x[test_count:])
    y_train = np.array(y[test_count:])
    x_test = np.array(x[:test_count])
    y_test = np.array(y[:test_count])

    return x_train, y_train, x_test, y_test
