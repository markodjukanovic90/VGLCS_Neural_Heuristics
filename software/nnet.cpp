#include "BeamSearch.h"
#include "globals.h"

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::uniform_real_distribution<double> standard_distribution_01(0.0,1.0);

// training GA parameters
int training_BS_time_limit = 10;
int training_BS_beta = 100;

Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
    return x.array().max(0);
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) { // use Eigen package for efficient matrix multiplication
    return 1.0 / (1.0 + (-x).array().exp());  
}

void apply_activation_function(Eigen::MatrixXd& x){
    
    if(activation_function == 1) x = x.array().tanh();
    else if(activation_function == 2) x = relu(x);
    else if(activation_function == 3) x = sigmoid(x);
}


int produce_random_integer(int max, double rval) {

    int num = int(double(max) * rval);
    if (num == max) num = num - 1;
    return num;
}

void print_information(double best_ofv, double ctime, int niter, double validation_value){
    
    cout << endl;
    cout <<"-------------------------------------------------------------------------------------------" << endl;
    cout << "best " << best_ofv << " time " << ctime << " niter: " << niter + 1 << " validation value: " << validation_value << endl;
    cout <<"-------------------------------------------------------------------------------------------" << endl;
}

MLP::MLP(){}

Eigen::VectorXd MLP::forward(const Eigen::VectorXd& x) {
    Eigen::MatrixXd x_mat = x;
    Eigen::MatrixXd prev = x_mat;
    for(int i = 0; i < (int) units_per_layer.size() - 1; ++i){
        Eigen::MatrixXd y = weight_matrices[i] * prev;
        y = y + bias_vectors[i];
        apply_activation_function(y);
        prev = y;
    }
    return prev;
}

void MLP::write_weights_to_file(const vector<double>& weights, double time){
    fstream weights_file;
    string filename = "weights_";
    filename += to_string(time);
    filename += ".txt";
    weights_file.open(filename,ios_base::out);
    for(double weight : weights)
        weights_file << weight << " ";

    weights_file.close();
}

double MLP::calculate_validation_value(const vector<double>& weights)
{ 
    
    this->store_weights(weights);
    double validation_value = 0;
    //thread  pool tp(thread::hardware_concurrency());
    for(Instance instance : validation_instances)
        validation_value += BeamSearch::imsbs_with_learning(&instance, training_beam_width, 10, 
                                        HeuristicType::H5, number_of_roots, 50, 10,  this ).best_seq.size(); // BS is run and the value obtained is accumulated
    
    validation_value = validation_value / validation_instances.size();
    return validation_value;
}

void MLP::apply_decoder(training_individual& ind){ //calculates the quality of the training individual
    this->store_weights(ind.weights);
    double ofv = 0;
    for(Instance instance : training_instances) 
    {
        //instance.print(std::cout);
        ofv += BeamSearch::imsbs_with_learning(&instance, training_beam_width, 10, HeuristicType::H5, number_of_roots, 
                                               50, 10,  this ).best_seq.size();
    }
        
    ofv = ofv / training_instances.size();
    ind.ofv = ofv;
}

void MLP::store_weights(const vector<double>& weights){ //store weights into neural network
    weight_matrices.clear();
    bias_vectors.clear(); 
    int stored_weights = 0;
    for(int i = 0; i < (int)units_per_layer.size() - 1; ++i){
        //weight matrix from layer i to layer i + 1
        int num_rows = units_per_layer[i + 1];
        int num_cols = units_per_layer[i];
        Eigen::MatrixXd weights_matrix(num_rows, num_cols);
        for (int i1 = 0; i1 < num_rows; ++i1) {
            for(int i2 = 0; i2 < num_cols; ++i2){
                weights_matrix(i1, i2) = weights[stored_weights];
                stored_weights ++;
            }
        }
        weight_matrices.push_back(weights_matrix);
        //bias vector from layer i to layer i + 1
        int num_biases = units_per_layer[i + 1];
        Eigen::MatrixXd bias_vector(num_biases, 1);
        for(int i1 = 0; i1 < num_biases; ++i1){
             bias_vector(i1) = weights[stored_weights];
             stored_weights ++;
        }
        bias_vectors.push_back(bias_vector);
    }
}

void MLP::write_training_and_validation_values(fstream& training_values_file, fstream& validation_values_file, double time, int niter, double training_value, double validation_value){
    //write training value to file
    training:Q_values_file << time << "\t" << niter << "\t" << training_value << endl;
    validation_values_file << time << "\t" << niter << "\t" << validation_value << endl;
    
}


vector<double> MLP::Train(){

    std::uniform_real_distribution<double> standard_distribution_weights(-weight_limit, weight_limit);
    
    int n_weights = 0; 
    for(int i = 0; i <(int) units_per_layer.size() - 1; ++i) 
        n_weights +=  (units_per_layer[i] + 1) * units_per_layer[i + :x!1];
    
    //results files
    fstream training_values_file;
    fstream validation_values_file;
    validation_values_file.open("validation_values.txt", ios_base::out);
    training_values_file.open("training_values.txt",ios_base::out);
    training_values_file << "Time" << "\t" << "Generations" << "\t" << "Training value" << endl;
    validation_values_file << "Time" << "\t" << "Generations" << "\t" << "Validation value" << endl;
    std::cout << "Time" << "\t" << "Generations" << "\t" << "Validation value" << std::endl;

    //time
    bool stop = false;
    auto start = std::chrono::steady_clock::now();
    double ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    int niter = 0;
    
    int n_offspring = population_size - n_elites - n_mutants;
    
    vector<training_individual> population(population_size);
    double best_ofv = std::numeric_limits<double>::min();
    vector<double> best_weights;
    std::cout <<"population... " << endl;
    for(int pi = 0; pi < population_size and not stop; ++pi){ //initialize every individual's chromosome as random
        
        population[pi].weights = vector<double>(n_weights);
        for(int i = 0; i < n_weights; i++)
            (population[pi].weights)[i] = standard_distribution_weights(generator);
        
        
        apply_decoder(population[pi]);
        
        ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if(ctime > training_time_limit) stop = true;
        if(population[pi].ofv > best_ofv){
            best_ofv = population[pi].ofv;
            best_weights = population[pi].weights;
            write_weights_to_file(best_weights, ctime);
            double validation_value = calculate_validation_value(best_weights);
            write_training_and_validation_values(training_values_file, validation_values_file, ctime, niter, best_ofv, validation_value);
            print_information(best_ofv, ctime, niter, validation_value);
        }
    }
    ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if(ctime > training_time_limit) stop = true;
   
    
    while(!stop){
    
        sort(population.begin(), population.end(), [](const training_individual& a, const training_individual& b) {return a.ofv > b.ofv;});
        
        vector<training_individual> new_population(population_size);
        for(int ic = 0; ic < n_elites; ++ic){ //construct elite population
            new_population[ic].weights = population[ic].weights;
            new_population[ic].ofv = population[ic].ofv;
        }
        
        for(int ic = 0; ic < n_mutants and not stop; ++ic){ //construct mutant population
            new_population[n_elites + ic].weights = vector<double>(n_weights);
            for(int i = 0; i < n_weights; ++i){
                (new_population[n_elites + ic].weights)[i] = standard_distribution_weights(generator);
            }
            apply_decoder(new_population[n_elites + ic]);
            ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if(ctime > training_time_limit) stop = true;

            if(new_population[n_elites + ic].ofv > best_ofv and not stop){
                best_ofv = new_population[n_elites + ic].ofv;
                best_weights = new_population[n_elites + ic].weights;
                write_weights_to_file(best_weights, ctime);
                double validation_value = calculate_validation_value(best_weights);
                write_training_and_validation_values(training_values_file, validation_values_file, ctime, niter, best_ofv, validation_value);
                print_information(best_ofv, ctime, niter, validation_value);
            }
        }
        
        for(int ic = 0; ic < n_offspring and not stop; ++ic){ //construct offspring population
        
            if(ga_config == 1){ // rkga offspring construction
                std::vector<int> indices(population_size); //shuffle all individuals and select first two as parents
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), generator);
                int first_parent = indices[0];
                int second_parent = indices[1];
                new_population[n_elites + n_mutants + ic].weights = vector<double>(n_weights);
                  for(int i = 0; i < n_weights; ++i){
                    double rnum = standard_distribution_01(generator);
                    if(rnum <= 0.5){
                        (new_population[n_elites + n_mutants + ic].weights)[i] = (population[first_parent].weights)[i];
                    }
                    else{
                        (new_population[n_elites + n_mutants + ic].weights)[i] = (population[second_parent].weights)[i];
                    }
                }
            }
            else if(ga_config == 2){ // brkga offspring construction
                double rval = standard_distribution_01(generator);
                int first_parent = produce_random_integer(n_elites, rval);
                rval = standard_distribution_01(generator);
                int second_parent = n_elites + produce_random_integer(population_size - n_elites, rval);
                new_population[n_elites + n_mutants + ic].weights = vector<double>(n_weights);
                for (int i = 0; i < n_weights; ++i) {
                    double rnum = standard_distribution_01(generator);
                    if (rnum <= elite_inheritance_probability) {
                        new_population[n_elites + n_mutants + ic].weights[i] = (population[first_parent].weights)[i];
                    }
                    else {
                        new_population[n_elites + n_mutants + ic].weights[i] = (population[second_parent].weights)[i];
                    }
                }
            }
            else if(ga_config == 3){ // lexicase selection for choosing parents
                vector<training_individual> parents;
                for(int i = 0; i < 2; ++i){ //choose the two parents
                    vector<Instance> shuffled_training_instances = training_instances;
                    std::shuffle(shuffled_training_instances.begin(), shuffled_training_instances.end(), generator);
                    vector<training_individual> considered_individuals;
                    for(Instance& instance : shuffled_training_instances){
                        double best_value = 0;
                        for(training_individual& individual : population){
                            store_weights(individual.weights);
                            double value = BeamSearch::imsbs_with_learning(&instance, training_beam_width, 10, 
                                                       HeuristicType::H5, number_of_roots, 50, 10,  this ).best_seq.size(); // it was instead od 100 and 60 --> 50, 10

                            if(value >= best_value){
                                if(value > best_value){
                                    best_value = value;
                                    considered_individuals.clear();
                                }
                                considered_individuals.push_back(individual);
                            }
                        }
                    }
                    double rval = standard_distribution_01(generator);
                    int selected = produce_random_integer(considered_individuals.size(), rval);
                    parents.push_back(considered_individuals[selected]);
                }

                new_population[n_elites + n_mutants + ic].weights = vector<double>(n_weights);
                for (int i = 0; i < n_weights; ++i) {
                    double rnum = standard_distribution_01(generator);
                    if (rnum <= 0.5) {
                        new_population[n_elites + n_mutants + ic].weights[i] = (parents[0].weights)[i];
                    }
                    else {
                        new_population[n_elites + n_mutants + ic].weights[i] = (parents[1].weights)[i];
                    }
                }

            }
            apply_decoder(new_population[n_elites + n_mutants + ic]);
            ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();


            if( ctime > training_time_limit ) stop = true;
            if(new_population[n_elites + n_mutants + ic].ofv > best_ofv and not stop){
            
                best_ofv = new_population[n_elites + n_mutants + ic].ofv;
                best_weights = new_population[n_elites + n_mutants + ic].weights;
                write_weights_to_file(best_weights, ctime);
                double validation_value = calculate_validation_value(best_weights);
                write_training_and_validation_values(training_values_file, validation_values_file, ctime, niter, best_ofv, validation_value);
                print_information(best_ofv, ctime, niter, validation_value);
            }
        }
    
        population.clear();
        population = new_population;
        ctime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if( ctime > training_time_limit ) stop = true;
        niter ++; std::cout << "Generation " << niter << " completed." << std::endl;
    }
    cout << "------------ END OF TRAINING ------------" << endl;
    
    validation_values_file.close();
    training_values_file.close();
    return best_weights; 
}


/*MLP MLP::make_model(size_t in_channels,
        size_t out_channels,
        size_t hidden_units_per_layer,
        int hidden_layers) {
    std::vector<size_t> units_per_layer;

    units_per_layer.push_back(in_channels);

    for (int i = 0; i < hidden_layers; ++i)
        units_per_layer.push_back(hidden_units_per_layer);

    units_per_layer.push_back(out_channels);

    MLP model(units_per_layer);
    return model;
}
*/

