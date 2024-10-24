import pandas as pd
import numpy as np
import random


# *****************************
# Nearest Neighbors Algorithm *
# *****************************

# This functions as a regular Nearest Neighbor algorithm if n = 1 and start is a random node.
# Also functions as a Repeated Randomized Nearest Neighbor with repeated calls with different start/n
def nearest_neighbor(matrix, n, start): 

    # setting up variables for algorithm and performance tracking
    visited = set()
    tour = []
    exp = 0     # number of expanded nodes
    cost = 0    # total cost

    # starting algorithm
    visited.add(start)
    tour.append(start)
    curr = start
    while len(tour) != len(matrix):

        # expanding curr node
        row = matrix.iloc[curr]
        exp += 1
        neighbors = []
        for node, n_cost in enumerate(row):
            if node not in visited:
                neighbors.append((n_cost, node))    # add the destination nodes and the costs to fringe

        # finding the n-closest neighbors and randomly choosing one
        n_closest = sorted(neighbors, key=lambda x: x[0])
        n_closest = n_closest[:n]
        new_node = random.choice(n_closest)     # if n is 1, this will just be 
                                                # the closest one (i.e. regular NN)

        # updating 
        curr = new_node[1]
        visited.add(curr)       # add it to visited
        cost += new_node[0]
        tour.append(curr)

    # finishing tour by returning to start
    cost += matrix.iloc[curr][start]
    tour.append(start)

    return cost, exp, tour

# **********************************
# Nearest Neighbor 2-Opt Algorithm *
# **********************************

# calculates the total distance of a tour
def calc_dist(tour, matrix):
    dist = 0
    for i in range(len(tour) - 1):
        dist += matrix.loc[tour[i], tour[i+1]]
    return dist

def nn2o(tour, matrix):
    # setting up variables
    best_tour = tour
    best_dist = calc_dist(tour, matrix)
    imp = True  # used to keep track of if the tour was improved

    while imp:
        imp = False
        for i in range(1, len(best_tour) - 2):          # loop through the nodes starting at the second
            for j in range(i + 1, len(best_tour) - 1):  # loop through the nodes after node i
                if j - i != 1:
                    new_tour = best_tour[:]     # copy best tour
                    new_tour[i:j] = reversed(best_tour[i:j])    # reverse segment from i to j-1
                    new_dist = calc_dist(new_tour, matrix)      
                    if new_dist < best_dist:    # if new dist is better -> keep it
                        best_tour = new_tour
                        best_dist = new_dist
                        imp = True

    return best_tour, best_dist

# *********************************************************
# Repeated Randomized Nearest Neighbor w/ 2-Opt Algorithm *
# *********************************************************


def rrnn2o(matrix, n, r):

    # initializing best tour starting at node 0
    best_cost, exp, best_tour = nearest_neighbor(matrix, n, 0)

    # repeat nn2o algorithm for all starting nodes
    for i in range(1, r):
        start = random.randint(0, len(matrix) - 1)
        new_cost, new_exp, new_tour = nearest_neighbor(matrix, n, start)
        new_tour, new_cost = nn2o(new_tour, matrix)
        exp += new_exp

        # if the cost is shorter then it becomes the new best tour
        if new_cost < best_cost:
            best_cost = new_cost
            best_tour = new_tour
            
    #best_tour, best_cost = nn2o(best_tour, matrix)
    
    return best_cost, exp, best_tour

