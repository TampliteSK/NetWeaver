// cost.cpp: A list of cost functions to use for the trainer

#include "cost.hpp"
#include <cmath>
#include <cassert>

double mean_square_error(double *y_est, double *y_target, uint16_t n_samples) {
    double cost = 0;
    for (uint16_t i = 0; i < n_samples; ++i) {
        cost += pow(y_est - y_target, 2.0);
    }
    return cost / n_samples;
}

// y_target should be either 0 or 1, and y_est is a probability between 0 and 1.
double binary_cross_entropy(double *y_est, double *y_target, uint16_t n_samples) {
    double cost = 0;
    for (uint16_t i = 0; i < n_samples; ++i) {
        assert(y_target[i] == 0 || y_target[i] == 1);
        assert(y_est[i] > 0 && y_est[i] < 1);
        cost += y_target[i] * log(y_est[i]) + (1 - y_target[i]) * log(1 - y_est[i]);
    }
    return -cost / n_samples;
}