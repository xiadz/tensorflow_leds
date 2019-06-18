#!/usr/bin/env python3

import csv
import math
import numpy as np
import random

INPUT_FILE="raw_nn_outputs.csv"
SORTED_COR_FILE="sorted_correlations.csv"
OPTIMIZATION_STEPS=500 * 1000
OPTIMIZED_ORDER_FILE="nn_outputs_order.csv"

print("Loading data")
outputs = np.genfromtxt(INPUT_FILE, delimiter=',')

num_rows, num_nn_outputs = outputs.shape
print("Num rows: %d" % num_rows)
print("Num NN outputs: %d" % num_nn_outputs)

print("Computing correlations")
correlations = np.corrcoef(outputs, rowvar=False)

print("Getting all correlation values")
values = []
for i in range(0, correlations.shape[0]):
    for j in range(i + 1, correlations.shape[1]):
        values.append(correlations[i, j])
values.sort()

print("Writing sorted correlations to %s" % SORTED_COR_FILE)
with open(SORTED_COR_FILE, "w") as f:
    for value in values:
        f.write(str(value))
        f.write("\n")

# Trying to reorder NN outputs so that correlated outputs
# are close to each other. This is, in essence, the travelling
# salesman problem.
# Using a simple genetic algorithm that swaps two outputs at a time.

outputs_map = np.zeros((16, 16, 3), dtype=np.int32)
i = 0
for y in range(16):
    for x in range(16):
        for c in range(3):
            outputs_map[y, x, c] = i
            i += 1


print("Trying to optimize")
def order_weight(order):
    def points_weight(point1, point2):
        return max(1.0 - correlations[order[point1], order[point2]], 0.0)

    weight = 0.0
    for y in range(16):
        for x in range(16):
            for c in range(3):
                this_point = outputs_map[y, x, c]
                if y > 0:
                    neighbor = outputs_map[y - 1, x, c]
                    weight += points_weight(this_point, neighbor)
                if y < 15:
                    neighbor = outputs_map[y + 1, x, c]
                    weight += points_weight(this_point, neighbor)
                if x > 0:
                    neighbor = outputs_map[y, x - 1, c]
                    weight += points_weight(this_point, neighbor)
                if x < 15:
                    neighbor = outputs_map[y, x + 1, c]
                    weight += points_weight(this_point, neighbor)
    return weight

order = random.sample(range(num_nn_outputs), num_nn_outputs);
initial_weight = order_weight(order)
print("Initial weight:", initial_weight)

weight = initial_weight
for step, temperature in enumerate(np.logspace(5.0, 0.0, num=OPTIMIZATION_STEPS)):
    if step % 1000 == 0:
        print("temperature: %.1f, order weight: %.1f" % (temperature, weight))
    i, j = sorted(random.sample(range(num_nn_outputs), 2))
    new_order =  order[:i] + order[j:j+1] +  order[i+1:j] + order[i:i+1] + order[j+1:];
    new_weight = order_weight(new_order)
    temp_change = (weight - new_weight) * temperature
    if temp_change > random.random():
        order = new_order
        weight = new_weight

print("Final weight:", weight)
with open(OPTIMIZED_ORDER_FILE, "w") as f:
    for i in order:
        f.write("%d\n" % i)

