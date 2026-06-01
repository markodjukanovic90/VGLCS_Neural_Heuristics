#include <iostream>
#include <fstream>
#include <sstream>
#include "BeamSearch.h"
#include "globals.h"


using namespace std;
MLP* neural_network = new MLP();
bool training = false;
int hidden_layers = 0;


void read_parameters(int argc,char** argv){
    
    int iarg = 1;
    std::vector<int> units;
    while (iarg < argc) {
    
        if (strcmp(argv[iarg], "-train") == 0) {
            
            // include instances for training 
            if(not training) training = true;
            std::string input_path = argv[++iarg];
            neural_network->training_instances.push_back( Instance::loadFromFile(input_path) );
        }
        // include instances for the validation phase
        else if (strcmp(argv[iarg], "-validation") == 0){
        
            std::string input_path = argv[++iarg];
            neural_network->validation_instances.push_back( Instance::loadFromFile(input_path) );
        }
        //weight limit pass: W \in [-a, a]
        else if (strcmp(argv[iarg], "-weight_limit") == 0) neural_network->weight_limit = atoi(argv[++iarg]);
        //pass the beam search for training (the one used in each forward pass); for the backward pass it is fixed to 10 
        else if (strcmp(argv[iarg], "-training_beam_width") == 0) neural_network->training_beam_width = atoi(argv[++iarg]);
        // time limit allowed for training
        else if (strcmp(argv[iarg],"-training_time_limit") == 0) neural_network->training_time_limit = atof(argv[++iarg]);
        //number of hidden layers in the neural network 
        else if (strcmp(argv[iarg], "-hidden_layers") == 0) hidden_layers = atoi(argv[++iarg]);
        // number of nodes in each layer: can be passed as -units 5 10 5, meaning 5 nodes in layer0, 10 nodes in layer1, 5 nodes in layer2
        else if (strcmp(argv[iarg], "-units") == 0){
            for(int i = 0; i < hidden_layers; ++i)
                units.push_back(atoi(argv[++iarg]));
        }
        // Basic Beam search
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
        else if (strcmp(argv[iarg], "-h") == 0 ) { // HEURISTIC FOR FORWARD BS in IMSBS
             heuristic = parseHeuristic(argv[++iarg]);  
        }
        else if ( strcmp(argv[iarg], "-m") == 0 ) {
             max_iters = std::stoi(argv[++iarg]);
        }
        else if ( strcmp(argv[iarg], "-nr") == 0 ) {
            number_of_roots = std::stoi(argv[++iarg]);
        }
        // activation function to pass 
        else if(strcmp(argv[iarg], "-activation_function") == 0){
            activation_function = atoi(argv[++iarg]);
        }
        // basic feature conf: 6 features, 
        else if(strcmp(argv[iarg], "-feature_configuration") == 0){
        
            feature_config = atoi(argv[++iarg]);  
            if(feature_config == 1) num_features = 9; // p^{L,v}, l^v and length of partial sol 
            else if(feature_config == 2) num_features = 10; // plus alphabet size 
            else if(feature_config == 3) num_features = 11; //plus num of input  strings
            else if(feature_config == 4) num_features = 12; //plus num of and length of input   strings
        }
        // Params. of the optimization algorithm (EA) used for tuning the NN (finding a suitable configuration)
        else if (strcmp(argv[iarg], "-ga_configuration") == 0){ //1: rkga, 2: brkga, 3: lexicase sel.
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
    //Set up neural network
    neural_network->units_per_layer.push_back(num_features); //input layer
    for(int i = 0; i < hidden_layers; ++i){ //inner layers
        std::cout << "Layer " << i << " with " << units[i] << " units " << std::endl;
        neural_network->units_per_layer.push_back(units[i]); // put nodes into (hidden) layers
    }
    neural_network->units_per_layer.push_back(1); //output layer
    
    if(not training){
        
        // include weights from file @weights.txt
        ifstream weights_file("weights.txt", std::ios::in);
        if(!weights_file.is_open()){
            cout << "Couldn't open the weights file" << endl;
            exit(-1);
        }
        vector<double> temporal_weights;
        double num = 0.0;  
        while(weights_file >> num) temporal_weights.push_back(num);
        neural_network->store_weights(temporal_weights);
    }
    // validation of params 
    if(feature_config != 1 and feature_config != 2 and feature_config != 3 and feature_config != 4){
        cout << "WARNING: feature configuration not specified. Using default value of one" << endl;
        feature_config = 1;
    }
    if(ga_config != 1 and ga_config != 2 and ga_config != 3 and training){
        cout << "WARNING: training ga configuration not specified. Using default value of one (rkga)" << endl;
        ga_config = 1;
    }
    if(neural_network->weight_limit == 0 and training){
        cout << "WARNING: weight limit not specified. Using default value of one" << endl;
        neural_network->weight_limit = 1;
    }
    if(neural_network->units_per_layer.size() == 0){
        cout << "ERROR: the shape of the neural network needs to be explicited " << endl;
        exit(-1);
    }
    if(activation_function != 1 and activation_function != 2 and activation_function != 3){
        cout << "WARNING: using no activation function, explicit one with -activation_function {1:tanh , 2:relu, 3:sigmoid}" << endl;
    }
}

int main( int argc, char **argv ) {

    read_parameters(argc, argv);
    if(training)
    { 
    
        std::cout << "Set up the training process  "  <<std::endl;
        std::cout << std::setprecision(10) << std::fixed;
        vector<double> res_weights = neural_network->Train(); // train NN 
        
        // write best weights inside weights file
        fstream weights_file;
        weights_file.open("last_weights.txt",ios_base::out);
        for(double weight : res_weights)
            weights_file << weight << " ";
        
        weights_file.close();
    }
    else { // if not trained, prioritize these nodes as outcome of the trained NN

           // ---------------- Select output stream ----------------
        std::ostream* out = &std::cout;
        std::ofstream fout;

        if (!outpathname.empty()) {
            fout.open(outpathname);
            if (!fout) {
                std::cerr << "Cannot open output file: " << outpathname << "\n";
                return 1;
            }
            out = &fout;
        }

        Instance inst = Instance::loadFromFile(input_path);
        //inst.print(std::cout);
        if(hidden_layers >  0) // runnig IMSBS with learning algorithm
        {   
            std::cout << "Run the learning IMSBS:  " << std::endl;
            BeamSearch::Result res_imsbs = BeamSearch::imsbs_with_learning(&inst, beam_width, 10, 
                                       heuristic, number_of_roots, 
                                       max_iters, time_limit_imsbs, neural_network) ;
            //Stats
            //Stats 
            BeamSearch::write_result(
                *out,
                res_imsbs,
                inst,
                "IMSBS-learning"
             ); 

        }else{ // running just basic IMSBS with no learning
            std::cout << "Run the basic IMSBS:  " << std::endl;
            BeamSearch::Result res_imsbs = BeamSearch::imsbs_with_learning(&inst, beam_width, 10, 
                                           heuristic, number_of_roots, 
                                           max_iters, time_limit_imsbs, nullptr);   
            //Stats 
            BeamSearch::write_result(
                *out,
                res_imsbs,
                inst,
                "IMSBS"
             );                                              
        }
        
        // save in file: delete instance after use
        //delete &inst;
    }
    return 0;
}

 