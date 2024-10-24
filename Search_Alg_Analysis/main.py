import os
import sys
import argparse
import time
import random
import pandas as pd
import numpy as np
import Search_H
import A_Star_MST
import Local_Search

# takes in the lines of stdin (starting at the second line) and returns a dataframe of the matrix
def read_matrix(lines):
    data = []
    for line in lines:
        row = list(map(float, line.split()))
        data.append(row)
    return pd.DataFrame(data)

def main(file_path):

    # reading infile
    try:
        with open(file_path, 'r') as file:
            n = file.readline()
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error reading the file: {e}")
    
    matrix = read_matrix(lines)
    #matrix = read_matrix(lines)

    # prints n and matrix just cuz
    print('N: ', n)
    print('Matrix: \n', matrix)

    # *******************
    # Nearest Neighbors *
    # *******************

    nn_rand_st = random.randint(0, len(matrix) - 1)

    nn_start_real = time.time()
    nn_start_cpu = time.process_time()
    nn_cost, nn_exp, nn_tour = Search_H.nearest_neighbor(matrix, 1, nn_rand_st)
    nn_end_real = time.time()
    nn_end_cpu = time.process_time()
    
    # **********************************
    # Nearest Neighbor 2-Opt Algorithm *
    # **********************************

    nn2o_start_real = time.time()
    nn2o_start_cpu = time.process_time()
    nn2o_tour, nn2o_cost = Search_H.nn2o(nn_tour, matrix)
    nn2o_end_real = time.time()
    nn2o_end_cpu = time.process_time()
    
    # *********************************************************
    # Repeated Randomized Nearest Neighbor w/ 2-Opt Algorithm *
    # *********************************************************

    rnn_start_real = time.time()
    rnn_start_cpu = time.process_time()
    rnn_cost, rnn_exp, rnn_tour = Search_H.rrnn2o(matrix, 1, 0)
    rnn_end_real = time.time()
    rnn_end_cpu = time.process_time()
    
    # *************
    # A* With MST *
    # *************

    A_start_real = time.time()
    A_start_cpu = time.process_time()
    A_cost, A_tour , A_exp = A_Star_MST.a_star(matrix)
    A_end_real = time.time()
    A_end_cpu = time.process_time()
    
    # ************
    # Hill Climb *
    # ************

    HC_start_real = time.time()
    HC_start_cpu = time.process_time()
    HC_tour, HC_cost, HC_exp = Local_Search.rrhc(matrix, min(len(matrix), 10))
    HC_end_real = time.time()
    HC_end_cpu = time.process_time()
    
     # *******************
    # Nearest Neighbors *
    # *******************

    SA_start_real = time.time()
    SA_start_cpu = time.process_time()
    SA_tour, SA_cost, SA_exp = Local_Search.rrsa(matrix, 100, 0.001, 0.99, 8)
    SA_end_real = time.time()
    SA_end_cpu = time.process_time()
    
     # *******************
    # Nearest Neighbors *
    # *******************

    G_start_real = time.time()
    G_start_cpu = time.process_time()
    G_tour, G_cost = Local_Search.gen(matrix, 100, 50, 0.6)
    G_end_real = time.time()
    G_end_cpu = time.process_time()
    
    
    

    print('-----------------')
    print('Nearest Neighbor |')
    print('-----------------')
    print('Tour: ', nn_tour)
    print('Best Cost: ', nn_cost)
    print('Number of Nodes Expanded: ', nn_exp, '\n')
    print('CPU Runtime', (nn_start_cpu - nn_end_cpu))
    print('Real Runtime', (nn_start_real - nn_end_real), '\n')
    print('----------------------------')
    print('Nearest Neighbor With 2-Opt |')
    print('----------------------------')
    print('Tour: ', nn2o_tour)
    print('Best Cost: ', nn2o_cost)
    print('Number of Nodes Expanded: ', nn_exp, '\n')
    print('CPU Runtime', (nn2o_start_cpu - nn2o_end_cpu))
    print('Real Runtime', (nn2o_start_real - nn2o_end_real), '\n')
    print('-------------------------------------------------')
    print('Reapeated Randomized Nearest Neighbor With 2-Opt |')
    print('-------------------------------------------------')
    print('Tour: ', rnn_tour)
    print('Best Cost: ', rnn_cost)
    print('Number of Nodes Expanded: ', rnn_exp, '\n')
    print('CPU Runtime', (rnn_start_cpu - rnn_end_cpu))
    print('Real Runtime', (rnn_start_real - rnn_end_real), '\n')
    print('---')
    print('A* |')
    print('---')
    print('Tour: ', A_tour)
    print('Best Cost: ', A_cost)
    print('Number of Nodes Expanded: ', A_exp, '\n')
    print('CPU Runtime', (A_start_cpu - A_end_cpu))
    print('Real Runtime', (A_start_real - A_end_real), '\n')
    print('-----------')
    print('Hill Climb |')
    print('-----------')
    print('Tour: ', HC_tour)
    print('Best Cost: ', HC_cost)
    print('Number of Nodes ExpHCnded: ', HC_exp, '\n')
    print('CPU Runtime', (HC_start_cpu - HC_end_cpu))
    print('real Runtime', (HC_start_real - HC_end_real), '\n')
    print('--------------------')
    print('Simulated Annealing |')
    print('--------------------')
    print('Tour: ', SA_tour)
    print('Best Cost: ', SA_cost)
    print('Number of Nodes Expanded: ', SA_exp, '\n')
    print('CPU Runtime', (SA_start_cpu - SA_end_cpu))
    print('Real Runtime', (SA_start_real - SA_end_real), '\n')
    print('---------')
    print('Genetics |')
    print('---------')
    print('Tour: ', G_tour)
    print('Best Cost: ', G_cost)
    print('CPU Runtime', (G_start_cpu - G_end_cpu))
    print('Real Runtime', (G_start_real - G_end_real), '\n')

    nn_data = {
        'Best Cost': [nn_cost],
        'Number of Nodes Expanded': [nn_exp],
        'CPU Runtime': [(nn_start_cpu - nn_end_cpu)],
        'Real Runtime': [(nn_start_real - nn_end_real)]
    }
    
    nn2o_data = {
        'Best Cost': [nn2o_cost],
        'Number of Nodes Expanded': [nn_exp],
        'CPU Runtime': [(nn2o_start_cpu - nn2o_end_cpu)],
        'Real Runtime': [(nn2o_start_real - nn2o_end_real)]
    }
    
    rnn_data = {
        'Best Cost': [rnn_cost],
        'Number of Nodes Expanded': [rnn_exp],
        'CPU Runtime': [(rnn_start_cpu - rnn_end_cpu)],
        'Real Runtime': [(rnn_start_real - rnn_end_real)]
    }
    
    A_data = {
        'Best Cost': [A_cost],
        'Number of Nodes Expanded': [A_exp],
        'CPU Runtime': [(A_start_cpu - A_end_cpu)],
        'Real Runtime': [(A_start_real - A_end_real)]
    }
    
    HC_data = {
        'Best Cost': [HC_cost],
        'Number of Nodes Expanded': [HC_exp],
        'CPU Runtime': [(HC_start_cpu - HC_end_cpu)],
        'Real Runtime': [(HC_start_real - HC_end_real)]
    }
    
    SA_data = {
        'Best Cost': [SA_cost],
        'Number of Nodes Expanded': [SA_exp],
        'CPU Runtime': [(SA_start_cpu - SA_end_cpu)],
        'Real Runtime': [(SA_start_real - SA_end_real)]
    }
    
    G_data = {
        'Best Cost': [G_cost],
        'CPU Runtime': [(G_start_cpu - G_end_cpu)],
        'Real Runtime': [(G_start_real - G_end_real)]
    }
    
    
    nn_df = pd.DataFrame(nn_data)
    nn_df.to_csv('CSV_Output/NN.csv', index=False)
    
    nn2o_df = pd.DataFrame(nn2o_data)
    nn2o_df.to_csv('CSV_Output/NN2O.csv', index=False)
    
    rnn_df = pd.DataFrame(rnn_data)
    rnn_df.to_csv('CSV_Output/RNN.csv', index=False)
    
    A_df = pd.DataFrame(A_data)
    A_df.to_csv('CSV_Output/A.csv', index=False)
    
    HC_df = pd.DataFrame(HC_data)
    HC_df.to_csv('CSV_Output/HC.csv', index=False)
    
    SA_df = pd.DataFrame(SA_data)
    SA_df.to_csv('CSV_Output/SA.csv', index=False)
    
    G_df = pd.DataFrame(G_data)
    G_df.to_csv('CSV_Output/Gen.csv', index=False)





if __name__ == '__main__':
     # Example usage
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a file path as a command line argument.")