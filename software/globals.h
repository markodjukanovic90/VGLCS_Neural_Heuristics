#ifndef GLOBAL_VARIABLE_H
#define GLOBAL_VARIABLE_H
#pragma once
#include <string>

// ====================== Parameters and Heuristic Types ======================

enum class HeuristicType {  
    H1,
    H2,
    H5,
    H8
};
 
//LBS features:
extern int num_features;

// IMSBS approach and its params 
extern HeuristicType heuristic; 
extern double beam_width;
extern int max_iters;
extern int time_limit_imsbs;
extern int number_of_roots;

// other params defaults:
extern std::string input_path;
extern std::string outpathname; // empty → stdout

// other parameters (necessary for tuning NN)
extern std::string outpathname;
extern std::string out_terminal_string;
extern std::string filename;
extern int activation_function;
extern int feature_config;
extern int population_size;
extern int num_threads;
//GA params: 
extern int n_elites;
extern int n_mutants;
extern int ga_config;
extern double elite_inheritance_probability;
extern bool differentiate_ties;
extern double lambda; // weight for the NN guidance in the fitness function of GA
extern int time_limit_imsbs; // time limit for the IMSBS algorithm in seconds (default: 30 mins)        
#endif // GLOBAL_VARIABLE_H
