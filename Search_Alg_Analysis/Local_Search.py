import pandas as pd
import numpy as np
import random

# ************
# Hill Climb *
# ************


def calc_cost(matrix, tour):
    cost = 0
    for i in range(len(tour) - 1):
        cost += matrix.iloc[tour[i], tour[i+1]]
    cost += matrix.iloc[tour[-1], tour[0]]
    return cost

def get_neighbors(tour):
    neighbors = []
    for i in range(1, len(tour) - 1):
        for j in range(i + 1, len(tour) - 1):
            neighbor = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


def hill_climb(matrix):
    tour = list(range(len(matrix)))
    random.shuffle(tour)
    tour = tour + [tour[0]]
    exp = 0
    
    curr_cost = calc_cost(matrix, tour)
    
    while True:
        neighbors = get_neighbors(tour)
        exp += len(neighbors)
        best_neighbor = None
        best_cost = curr_cost
        
        for neighbor in neighbors:
            n_cost = calc_cost(matrix, neighbor)
            if n_cost < best_cost:
                best_neighbor = neighbor
                best_cost = n_cost
                
        if best_cost >= curr_cost:
            break
        
        tour = best_neighbor
        curr_cost = best_cost
        
                
    return tour, curr_cost, exp

def rrhc(matrix, r):
    best_tour = None
    best_cost = float('inf')
    exp = 0
    
    for _ in range(r):
        tour, cost, nodes_exp = hill_climb(matrix)
        exp += nodes_exp
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
    
    return best_tour, best_cost, exp

# ********************
# Simulated Anealing *
# ********************

def sim_anneal(matrix, temp, min_temp, alpha):
    tour = list(range(len(matrix)))
    random.shuffle(tour)
    tour = tour + [tour[0]]
    exp = 0
    
    best_cost = calc_cost(matrix, tour)
    
    while temp > min_temp:
        # getting the neighbors
        neighbors = get_neighbors(tour)
        exp += len(neighbors)
        # Select a random neighbor
        random_neighbor = random.choice(neighbors)
        # Calculating cost and probability to select it
        n_cost = calc_cost(matrix, random_neighbor)
        if n_cost <= best_cost:
            tour = random_neighbor
            best_cost = n_cost
        else:
            probability = np.exp(-((n_cost) - (best_cost)) / temp)
            if probability > random.random():  # Accept with a certain probability
                tour = random_neighbor
                best_cost = n_cost
                
        # decreasing temp
        temp = temp * alpha
    
    return tour, best_cost, exp

def rrsa(matrix, temp, min_temp, alpha, r):
    
    best_tour = None
    best_cost = float('inf')
    exp = 0
    
    for _ in range(r):
        tour, cost, nodes_exp = sim_anneal(matrix, temp, min_temp, alpha)
        exp += nodes_exp
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
    
    return best_tour, best_cost, exp

# *********
# Genetic *
# *********

# initializes a random population
def init_pop(size, num_cities):
    pop = []
    for _ in range(size):
        tour = list(np.random.permutation(num_cities))
        pop.append(tour)
    return pop

# Fitness Function
# Picks a randome sample of size num_selected from a list of (tour, cost)
def fit_func(pop, cost, num_selected):
    selected = random.sample(list(zip(pop, cost)), num_selected)
    # returns the path with the lowest cost from the random sample
    return min(selected, key=lambda x: x[1])[0] 

# Crossover Function
# Creates crossovers of the 2 parents 
def crossover(p1, p2):
    size = len(p1)
    c1 = [-1] * size
    c2 = [-1] * size
    
    # choosing a random split point for the crossover (random crossover length)
    split = random.randint(1, size)
    # copying segment to child
    c1[:split] = p1[:split]
    c2[:split] = p2[:split]
    # Filling in the rest from other parent  
    def fill_rest(c, p):
        pos = split
        for city in p:
            if city not in c:
                if pos >= size:
                    pos = 0
                c[pos] = city
                pos += 1
    
    fill_rest(c1, p2)
    fill_rest(c2, p1)
    
    return c1, c2

# Mutation Function
# possibly swaps 2 random points in the tour
def mut(tour, mut_rate):
    if random.random() < mut_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def gen(matrix, size, gens, mut_rate, ):
    pop = init_pop(size, len(matrix))
    best_tour = None
    best_cost = float('inf')
    
    for _ in range(gens):
        costs = [calc_cost(matrix, tour)for tour in pop]
        new_pop = []
        
        for _ in range(size):
            p1 = fit_func(pop, costs, 1)
            p2 = fit_func(pop, costs, 1)
            
            c1, c2 = crossover(p1, p2)
            c1 = mut(c1, mut_rate)
            c2 = mut(c2, mut_rate)
            new_pop.append(c1)
            new_pop.append(c2)
            
        # Calculate the cost for the new offspring
        new_costs = [calc_cost(matrix, tour) for tour in new_pop]
        
        # Combine the old population with the new offspring
        pop += new_pop
        costs += new_costs
        
        # Sort population by cost
        sorted_pop = [tour for _, tour in sorted(zip(costs, pop))]
        sorted_costs = sorted(costs)
        
        # Keep only the best half of the population
        pop = sorted_pop[:size]
        costs = sorted_costs[:size]
        
        # Track the best solution in the population
        for i, tour in enumerate(pop):
            cost = costs[i]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
    
    return best_tour, best_cost