#include "BeamSearch.h"
#include "globals.h"

#include <omp.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>

using namespace std;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

thread_local std::default_random_engine generator(seed + omp_get_thread_num());

std::uniform_real_distribution<double> standard_distribution_01(0.0,1.0);

// training GA parameters
int training_BS_time_limit = 10;
int training_BS_beta = 100;

Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
    return x.array().max(0);
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
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

    Eigen::MatrixXd prev = x;

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

    //#pragma omp parallel for reduction(+:validation_value) schedule(dynamic)
    for(int i = 0; i < (int)validation_instances.size(); i++)
    {
        Instance& instance = validation_instances[i];

        validation_value += BeamSearch::imsbs_with_learning(
            &instance,
            training_beam_width,
            10,
            HeuristicType::H5,
            number_of_roots,
            50,
            10,
            this
        ).best_seq.size();
    }

    validation_value /= validation_instances.size();

    return validation_value;
}

void MLP::apply_decoder(training_individual& ind){

    this->store_weights(ind.weights);
    
    double ofv = 0;

    //#pragma omp parallel for reduction(+:ofv) schedule(dynamic)
    for(int i = 0; i < (int)training_instances.size(); i++)
    {
        Instance& instance = training_instances[i];

        ofv += BeamSearch::imsbs_with_learning(
            &instance,
            training_beam_width,
            10,
            HeuristicType::H5,
            number_of_roots,
            50,
            10,
            this
        ).best_seq.size();
    }

    ind.ofv = ofv / training_instances.size(); 
    // change ind.ofv  this for something more advanced:  ofv / training_instances.size() + penalty * 20-percentile results among all train_individuals, or something like that, to increase the pressure on the training to find weights that perform well on all training instances, not just a subset of them
    // ind.ofv = ofv / training_instances.size() + penalty * percentile_20;
    //percentile_20 can be calculated by collecting the results of all training individuals on all training instances, and then calculating the 20th percentile of these results. The penalty can be calculated as the proportion of training instances for which the individual's performance is below the 20th percentile. This way, individuals that perform well on a larger portion of the training instances will have a higher fitness value, even if their average performance is not the best.
    
    
}

void MLP::store_weights(const vector<double>& weights){

    weight_matrices.clear();
    bias_vectors.clear(); 

    int stored_weights = 0;

    for(int i = 0; i < (int)units_per_layer.size() - 1; ++i){

        int num_rows = units_per_layer[i + 1];
        int num_cols = units_per_layer[i];

        Eigen::MatrixXd weights_matrix(num_rows, num_cols);

        for (int i1 = 0; i1 < num_rows; ++i1)
            for(int i2 = 0; i2 < num_cols; ++i2)
                weights_matrix(i1, i2) = weights[stored_weights++];

        weight_matrices.push_back(weights_matrix);

        int num_biases = units_per_layer[i + 1];

        Eigen::MatrixXd bias_vector(num_biases, 1);

        for(int i1 = 0; i1 < num_biases; ++i1)
            bias_vector(i1) = weights[stored_weights++];

        bias_vectors.push_back(bias_vector);
    }
}

void MLP::write_training_and_validation_values(
    fstream& training_values_file,
    fstream& validation_values_file,
    double time,
    int niter,
    double training_value,
    double validation_value)
{
    training_values_file << time << "\t" << niter << "\t" << training_value << endl;
    validation_values_file << time << "\t" << niter << "\t" << validation_value << endl;
}

vector<double> MLP::Train(){

    std::uniform_real_distribution<double> standard_distribution_weights(-weight_limit, weight_limit);

    int n_weights = 0;

    for(int i = 0; i <(int) units_per_layer.size() - 1; ++i) 
        n_weights +=  (units_per_layer[i] + 1) * units_per_layer[i + 1];

    fstream training_values_file;
    fstream validation_values_file;

    validation_values_file.open("validation_values.txt", ios_base::out);
    training_values_file.open("training_values.txt",ios_base::out);

    training_values_file << "Time\tGenerations\tTraining value" << endl;
    validation_values_file << "Time\tGenerations\tValidation value" << endl;

    bool stop = false;

    auto start = std::chrono::steady_clock::now();
    double ctime;

    int niter = 0;

    int n_offspring = population_size - n_elites - n_mutants;

    vector<training_individual> population(population_size);

    double best_ofv = std::numeric_limits<double>::min();
    double best_validation_value = std::numeric_limits<double>::min();

    vector<double> best_weights;
    vector<double> best_weights_validation;

    const double epsilon = 1e-4;

    // -------- Parallel population initialization ------
    //#pragma omp parallel for schedule(dynamic)
    for(int pi = 0; pi < population_size; ++pi){

        population[pi].weights = vector<double>(n_weights);

        for(int i = 0; i < n_weights; i++)
            population[pi].weights[i] = standard_distribution_weights(generator);

        apply_decoder(population[pi]);
    }
    
    for(int pi = 0; pi < population_size; ++pi)
    {
        if(population[pi].ofv > best_ofv){
            best_ofv = population[pi].ofv;
            best_weights = population[pi].weights;
        }
    }

    // Initial validation evaluation
    double validation_value = calculate_validation_value(best_weights);
    best_validation_value = validation_value;
    best_weights_validation = best_weights;

    ctime = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    
    write_training_and_validation_values(training_values_file, validation_values_file, ctime, 0, best_ofv, validation_value);
    write_weights_to_file(best_weights_validation, ctime);
    //std::cout << "while ... ";
    while(!stop){

        sort(population.begin(), population.end(),
             [](const training_individual& a, const training_individual& b)
             {return a.ofv > b.ofv;});

        vector<training_individual> new_population(population_size);
         // BRKGA for training 
        // ---- elites ----
        for(int ic = 0; ic < n_elites; ++ic){
            new_population[ic] = population[ic];
        }

        // ---- mutants ----
        for(int ic = 0; ic < n_mutants; ++ic){

            new_population[n_elites + ic].weights = vector<double>(n_weights);

            for(int i = 0; i < n_weights; ++i)
                new_population[n_elites + ic].weights[i] = standard_distribution_weights(generator);
        }

        // ---- offspring ----
        for(int ic = 0; ic < n_offspring; ++ic){

            std::vector<int> indices(population_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), generator);

            int p1 = indices[0];
            int p2 = indices[1];

            new_population[n_elites + n_mutants + ic].weights = vector<double>(n_weights);

            for(int i = 0; i < n_weights; ++i){

                double rnum = standard_distribution_01(generator);

                if(rnum <= elite_inheritance_probability)
                    new_population[n_elites + n_mutants + ic].weights[i] = population[p1].weights[i];
                else
                    new_population[n_elites + n_mutants + ic].weights[i] = population[p2].weights[i];
            }
        }

        // -------- Parallel evaluation --------
        //#pragma omp parallel for schedule(dynamic)
        for(int i = n_elites; i < population_size; i++)
            apply_decoder(new_population[i]);

        // -------- Selection with validation control --------
        for(int i = 0; i < population_size; i++)
        {
            if(new_population[i].ofv > best_ofv)
            {
                double candidate_train = new_population[i].ofv;
                const vector<double>& candidate_weights = new_population[i].weights;

                double candidate_validation = calculate_validation_value(candidate_weights);

                if(candidate_validation >= best_validation_value - epsilon) //validation control: only accept new weights if validation value is not worse than the best validation value found so far (with a small tolerance epsilon)
                {
                    best_ofv = candidate_train;
                    best_validation_value = candidate_validation;

                    best_weights = candidate_weights;
                    best_weights_validation = candidate_weights;

                    ctime = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start).count();

                    write_weights_to_file(best_weights_validation, ctime);

                    write_training_and_validation_values(
                        training_values_file,
                        validation_values_file,
                        ctime,
                        niter,
                        best_ofv,
                        best_validation_value);

                    print_information(best_ofv, ctime, niter, best_validation_value);
                }
                else{ // te overfitting occured, STOP the training 
                    
                    stop = true;
                
                }
            }
        }  

        population = new_population;

        ctime = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();

        if(ctime > training_time_limit)
            stop = true;

        niter++;

        cout << "Generation " << niter << " completed." << endl;
    }

    cout << "------------ END OF TRAINING ------------" << endl;

    validation_values_file.close();
    training_values_file.close();

    // Return best VALIDATION weights (important!)
    return best_weights_validation;

}