#!/usr/bin/python
# -*- coding: utf-8 -*-
from functools import lru_cache
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

import numpy as np

def dp_select(values, weights, capacity):
    """Use dynamic programming to build up to solution for all items"""
    
    # convert params to n,K notation as used in equations in dynamic programming notes
    n = len(values)
    K = capacity
    
    # calculate table of optimal value by j,k (j items in 0..n, k is all capacities in 0..K)
    # see 9:00 - 12:30 here: https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming
    values_table = np.zeros((K+1,n+1), dtype=np.uint32)
    
    print("building DP optimal-value table for n,K: ", n, ",", K)
    for j in range(1, n+1):
        item_weight = weights[j-1]
        item_value  = values[j-1]
        for k in range(1, K+1):
            if item_weight > k:
                values_table[k,j] = values_table[k, j-1]
            else:
                values_table[k,j] = max(values_table[k, j-1], item_value + values_table[k-item_weight, j-1])
    optimal_value = values_table[-1, -1]
    print(f"optimal value is {optimal_value}. Now proceeding to derive final item-set")

    # from this table of optimal values, we now need to derive final item-set for optimal solution
    # logic of code below explained 12:30 - 14:00 at https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming
    taken = [0] * len(values)
    k = K   # in keeping w/ eqs, K is total capacity but k is k'th row as we move through j,k table
    for j in range(n, 0, -1):
        if values_table[k,j] != values_table[k,j-1]:
            taken[j-1] = 1
            k = k - weights[j-1]
    
    return optimal_value, taken


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    try:
        value, taken = dp_select([i.value for i in items], [i.weight for i in items], capacity)
    except:
        value = capacity
        taken = [0] * len(items)
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def knapSack(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    # Build tаble K[][] in bоttоm uр mаnner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0  or  w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1]
                        + K[i-1][w-wt[i-1]],
                            K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    taken = [0] * len(wt)
    trace_cap = W
    idx = n
    while n > 0:
        while K[n][trace_cap] == K[n-1][trace_cap] and n >=0:
            n -= 1
        taken[n-1] = 1
        
        trace_cap -= wt[n-1]
        n-=1
    return K[idx][W], taken
        
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

