#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <omp.h>

#include "BeamSearch.h"
#include "globals.h"

using namespace std;

MLP* neural_network = new MLP();
bool training = false;
int hidden_layers = 0;
//int num_threads = 1;   // NEW: number of threads


void read_parameters(int argc,char** argv){
    
    int iarg = 1;
    std::vector<int> units;

    while (iarg < argc) {
    
        if (strcmp(argv[iarg], "-train") == 0) {
            
            if(!training) training = true;

            std::string input_path = argv[++iarg];
            neural_network->training_instances.push_back(
                Instance::loadFromFile(input_path));
        }

        else if (strcmp(argv[iarg], "-validation") == 0){
        
            std::string input_path = argv[++iarg];
            neural_network->validation_instances.push_back(
                Instance::loadFromFile(input_path));
        }

        else if (strcmp(argv[iarg], "-threads") == 0){
            num_threads = atoi(argv[++iarg]);   // NEW
        }

        else if (strcmp(argv[iarg], "-weight_limit") == 0)
            neural_network->weight_limit = atoi(argv[++iarg]);

        else if (strcmp(argv[iarg], "-training_beam_width") == 0)
            neural_network->training_beam_width = atoi(argv[++iarg]);

        else if (strcmp(argv[iarg],"-training_time_limit") == 0)
            neural_network->training_time_limit = atof(argv[++iarg]);

        else if (strcmp(argv[iarg], "-hidden_layers") == 0)
            hidden_layers = atoi(argv[++iarg]);

        else if (strcmp(argv[iarg], "-units") == 0){
            for(int i = 0; i < hidden_layers; ++i)
                units.push_back(atoi(argv[++iarg]));
        }

        else if (strcmp(argv[iarg], "-timsbs") == 0){
            time_limit_imsbs = atoi(argv[++iarg]);
        }

        else if (strcmp(argv[iarg], "-i") == 0){
            input_path = argv[++iarg];
        }

        else if (strcmp(argv[iarg], "-o") == 0){
            outpathname = argv[++iarg];
        }

        else if (strcmp(argv[iarg], "-b") == 0){
            beam_width = atoi(argv[++iarg]);
        }

        else if (strcmp(argv[iarg], "-h") == 0){
             heuristic = parseHeuristic(argv[++iarg]);
        }

        else if ( strcmp(argv[iarg], "-m") == 0 ){
             max_iters = std::stoi(argv[++iarg]);
        }

        else if ( strcmp(argv[iarg], "-nr") == 0 ){
            number_of_roots = std::stoi(argv[++iarg]);
        }
        else if ( strcmp(argv[iarg], "-lambda") == 0 ){
            lambda = std::atof(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-activation_function") == 0){
            activation_function = atoi(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-feature_configuration") == 0){
        
            feature_config = atoi(argv[++iarg]);

            if(feature_config == 1) num_features = 9;
            else if(feature_config == 2) num_features = 10;
            else if(feature_config == 3) num_features = 11;
            else if(feature_config == 4) num_features = 12;
        }

        else if (strcmp(argv[iarg], "-ga_configuration") == 0){
            ga_config = atoi(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-population_size") == 0){
            population_size = atoi(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-n_elites") == 0){
            n_elites = atoi(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-n_mutants") == 0){
            n_mutants = atoi(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-rho") == 0){
            elite_inheritance_probability = atof(argv[++iarg]);
        }

        else if(strcmp(argv[iarg], "-differentiate_ties") == 0){
            differentiate_ties = true;
        }

        iarg++;
    }

    // Setup neural network
    neural_network->units_per_layer.push_back(num_features);

    for(int i = 0; i < hidden_layers; ++i){
        //cout << "Layer " << i << " with " << units[i] << " units" << endl;
        neural_network->units_per_layer.push_back(units[i]);
    }

    neural_network->units_per_layer.push_back(1);

    if(!training){
        
        ifstream weights_file("weights.txt");

        if(!weights_file.is_open()){
            cout << "Couldn't open the weights file" << endl;
            exit(-1);
        }

        vector<double> temporal_weights;
        double num;

        while(weights_file >> num)
            temporal_weights.push_back(num);

        neural_network->store_weights(temporal_weights);
    }

    if((feature_config < 1 || feature_config > 4) and hidden_layers  > 0){
        cout << "WARNING: feature configuration not specified. Using default value 1" << endl;
        feature_config = 1;
    }

    if(ga_config < 1 || ga_config > 3){
        if(training){
            cout << "WARNING: GA configuration not specified. Using rkga." << endl;
            ga_config = 1;
        }
    }

    if(neural_network->weight_limit == 0 && training){
        cout << "WARNING: weight limit not specified. Using 1." << endl;
        neural_network->weight_limit = 1;
    }

    if(neural_network->units_per_layer.size() == 0){
        cout << "ERROR: neural network shape must be specified." << endl;
        exit(-1);
    }

    if((activation_function < 1 || activation_function > 3) and hidden_layers > 0){
        cout << "WARNING: no activation function specified." << endl;
    }
}


int main( int argc, char **argv ) {

    read_parameters(argc, argv);

    // SET NUMBER OF THREADS
    //omp_set_num_threads(num_threads);

    //cout << "Using " << omp_get_max_threads() << " threads." << endl;

    if(training)
    { 
        cout << "Set up the training process" << endl;
        cout << std::setprecision(10) << std::fixed;

        vector<double> res_weights = neural_network->Train();

        fstream weights_file;
        weights_file.open("last_weights.txt",ios_base::out);

        for(double weight : res_weights)
            weights_file << weight << " ";

        weights_file.close();
    }
    else {

        ostream* out = &cout;
        ofstream fout;

        if (!outpathname.empty()) {
            fout.open(outpathname);

            if (!fout) {
                cerr << "Cannot open output file: " << outpathname << "\n";
                return 1;
            }

            out = &fout;
        }

        Instance inst = Instance::loadFromFile(input_path);

        if(hidden_layers > 0)
        {   
            cout << "Run the learning IMSBS" << endl;

            BeamSearch::Result res =
                BeamSearch::imsbs_with_learning(
                    &inst,
                    beam_width,
                    10,
                    heuristic,
                    number_of_roots,
                    max_iters,
                    time_limit_imsbs,
                    neural_network);

              BeamSearch::write_result(*out, res, inst, "IMSBS-learning");
              cout << res.best_seq.size();         
	}
        else
        {
            //cout << "Run the basic IMSBS" << endl;

            BeamSearch::Result res =
                BeamSearch::imsbs_with_learning(
                    &inst,
                    beam_width,
                    10,
                    heuristic,
                    number_of_roots,
                    max_iters,
                    time_limit_imsbs,
                    nullptr);

          BeamSearch::write_result(*out, res, inst, "IMSBS");
          cout << res.best_seq.size();
	}
    }

    return 0;
}
