// globals.cpp

#include "globals.h"


// IMSBS approach and its params 
HeuristicType heuristic = HeuristicType::H5; 
double beam_width = 10;
int max_iters = 10;
int time_limit_imsbs = 1800;
int number_of_roots = 10;

//INPUT and OUTPUT files 
std::string outpathname="";
std::string input_path="";

std::string out_terminal_string=""; 
int num_features = 9; //number of features
std::string filename;
int activation_function = 0; // activation function used
int feature_config = 0; // features used
int num_threads = 1;
int ga_config = 0; //type of GA used for training (rkga, brkga, lexicase selection)
//training rkga/brkga parameters
int population_size = 20;
int n_elites = 1;
int n_mutants = 7;
double elite_inheritance_probability = 0.5; //brkga rho
bool differentiate_ties = false; //whether to use the probabilistic heuristic as a secondary heuristic function

double lambda = 0.45; // weight for the NN guidance in the fitness function of GA     
