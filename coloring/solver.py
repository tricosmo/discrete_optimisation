#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx

import numpy as np


def color_graph(edges, extra_iterations=5):
    """Welsh Powell (Greedy) Algorith to color graphs + recoloring at the end.
    
    Parameters
    ----------
    edges : list of tuples of shape (n_edges, )
        List of of edges.
    extra_iterations : int
        How many iterations of recoloring will the algorithm do.
    Returns
    -------
    solution : list of shape (n_points, )
        List of colors (numbers) for every point in the graph.
    """

    edges = np.asarray(edges)
    vertex_count = np.amax(edges) + 1

    # Create the graph matrix
    graph = np.zeros(shape=(vertex_count, vertex_count), dtype=np.int8)
    rows, cols = zip(*edges)
    graph[rows, cols] = 1
    graph[cols, rows] = 1
    del rows, cols, edges

    # Gets the degree of every vertex and sorts them in a descending order
    colsum = np.argsort(-graph.sum(axis=0), kind="stable")

    colors = np.full(len(graph), -1, dtype=np.int8)
    for index in colsum:
        for color in range(vertex_count):
            for i, node in enumerate(graph[index]):
                if node == 1 and colors[i] == color and i != index:
                    break
            else:
                colors[index] = color
                break

    # Recolor TODO: Use Kemp Chains
    for p in range(extra_iterations):
        max_colores = colors.max()
        for index in colsum:
            adjacent_colors = {
                colors[i] for i, node in enumerate(graph[index]) if node == 1
            }
            for color in range(max_colores):
                if (
                    color != colors[index]
                    and color not in adjacent_colors
                    and color != max_colores
                ):
                    colors[index] = color
                    break

    return colors

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    colors = color_graph(edges)
    # print(colors)
    node_count = max(colors) + 1
    # build a trivial solution
    # every node has its own color
    solution = colors

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

