import pandas as pd
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import heapq


# This computes h(n) using MST
def mst_h(matrix, unvisited):   
    if len(unvisited) == 0:
        return 0
    
    # Creating matrix of unvisited
    cities = list(unvisited)
    sub = matrix.iloc[cities, cities]
    
    # MST for sub-matrix
    mst = minimum_spanning_tree(sub)
    cost = mst.sum()
    
    return cost

def a_star(matrix):
    #start, h_n, unvisited = find_start(matrix)
    start = 0
    unvisited = set(matrix.index) - {0}
    fringe = []
    # min-heap priority queue (f(n), current node, unvisited, g(n) (cost), path)
    heapq.heappush(fringe, (0 + mst_h(matrix, unvisited), start, unvisited, 0, [start]))
    exp = 1
    
    
    while fringe:
        # popping node with lowest f(n)
        f_n, curr, unvisited, g_n, path = heapq.heappop(fringe)
        
        # if we pop a goal node -> return
        if curr == start and len(unvisited) == 0:
            return f_n, path, exp
        
        # if no more nodes need to be visited, add start to end and push onto queue
        if len(unvisited) == 0:
            new_g_n = g_n + matrix.iloc[curr, start]
            new_path = path + [start]
            heapq.heappush(fringe, (new_g_n, start, unvisited, new_g_n, new_path))
        
        # add neighbors to fringe
        for neighbor in unvisited:
            exp += 1
            new_unvisited = unvisited - {neighbor}
            new_g_n = g_n + matrix.iloc[curr, neighbor]
            new_path = path + [neighbor]
            # Push new state onto fringe
            heapq.heappush(fringe, (new_g_n + mst_h(matrix, new_unvisited), neighbor, new_unvisited, new_g_n, new_path))
    
    return None

