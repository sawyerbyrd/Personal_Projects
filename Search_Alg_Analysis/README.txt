Hello to whoever see's my project! 

The purpose of this project was to implement and run tests on various search algorithms 
in order to gain insight on them.
The testing, outcomes, data, graphs, and conclusions are all 
inside the Report.ipynb file (There is also an HTML copy).

Bellow is some info on what is in this file and how to use the program.

This file contains 3 files for algorithm implementations and 1 report file:
    - Search_H contains three search hueristic algorithms
        -- NN (Nearest Neighbor)
        -- NN2O (Nearest Neighbor with 2-Opt)
        -- RNN  (Repeated Randomized Nearest Neighbor)
    - A_Star_MST contains the A* algorithm
    - Local_Search contains the three local search algorithms
        -- HC   (Hill Climb)
        -- SA   (Simulated Anealing)
        -- Gen  (Genetic algorithm)
    - Report contains my experiments, data graphs, and conclusions on all of the above algorithms
        -- there is also an HTML file of the report in here as well

To use my program, just type "python3 main.py path/to/input_file.txt"
    - there is a group of unit square graphs in the file titled "revised_square_graph_5_10_..._50_" 
        that can be used when runing the main file
    - The program will then print out the results of each algorithm in 
        the terminal, as well as an individual CSV for each algorithm
    