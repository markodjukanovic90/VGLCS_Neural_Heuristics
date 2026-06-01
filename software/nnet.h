#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <tuple>
#include <functional>
#include <random>
#include <cassert>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <set>
#include <map>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <chrono>
#include <Eigen/Dense>
#include "Instance.h"

using namespace std;

struct training_individual {
    vector<double> weights; //individual chromosome, corresponding to a set of weights.
    double ofv; //measure of quality of the weights, calculated by running BS guided by a NN with the weights of the individual for a set of training instances
};

class MLP {
    public:
        vector<size_t> units_per_layer;
        vector<Eigen::MatrixXd> bias_vectors;
        vector<Eigen::MatrixXd> weight_matrices;
        //vector<Eigen::MatrixXd> activations;
        vector<Instance> training_instances;
        vector<Instance> validation_instances;
        int training_beam_width;
        double training_time_limit;
        double weight_limit; //either use a weight limit or do some trick to not limit weights

        MLP();
        static inline double d_tanh(double x);
        Eigen::VectorXd forward(const Eigen::VectorXd& x);
        void write_weights_to_file(const std::vector<double>& weights, double time);
        double calculate_validation_value(const vector<double>& weights);
        vector<double> Train();
        void write_training_and_validation_values(fstream& training_values_file, fstream& validation_values_file, double time, int niter, double training_value, double validation_value);
        void apply_decoder(training_individual& ind);
        void store_weights(const vector<double>& weights);

        static MLP make_model(size_t in_channels,
        size_t out_channels,
        size_t hidden_units_per_layer,
        int hidden_layers);

};



