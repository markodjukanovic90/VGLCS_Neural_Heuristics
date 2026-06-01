#ifndef UTILS_H
#define UTILS_H
#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <limits>
#include <iostream>
#include <stdexcept>


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size())
            os << ", ";
    }
    os << "]";
    return os;
}

double compute_average(const std::vector<double>& v);
double compute_max(const std::vector<double>& v);
double compute_min(const std::vector<double>& v);
double compute_std(const std::vector<double>& v, double mean);
void standardize(std::vector<double>& v);

// ====================== Heuristic helpers ======================

HeuristicType parseHeuristic(const std::string& h);
std::string heuristicToString(HeuristicType h);

// ====================== Feasibility ======================

bool check_feasibility(
    const std::vector<std::vector<int>>& trace,
    const std::vector<std::vector<int>>& gaps,
    bool verbose = false
);

// ====================== Lower / Upper bounds ======================

int remaining_lb(
    const std::vector<int>& pos,
    const std::vector<std::string>& sequences
);

int h5_upper_bound(
    const std::vector<int>& pos,
    const std::vector<std::unordered_map<char, std::vector<int>>>& C_h5,
    const std::vector<char>& Sigma_h5,
    const std::vector<std::string>& sequences
);

// ====================== BACKWARD h5 ======================

int h5_backward(
    const std::vector<int>& pos,
    const std::vector<std::unordered_map<char, std::vector<int>>>& C_h5,
    const std::vector<char>& Sigma_h5
);

// ====================== Probability heuristic ======================

double probability_based_heuristic(
    const std::vector<int>& pos,
    const std::vector<std::vector<double>>& P,
    const std::vector<std::string>& sequences,
    int k
);

// ====================== Helpers ======================

std::unordered_map<char, int> first_valid_indices(
    const std::string& seq,
    int start_idx,
    const std::vector<int>& gap_arr,
    int prev_idx
);

#endif // UTILS_H
