// cost.hpp

#ifndef COST_HPP
#define COST_HPP

#include <cstdint>

double mean_square_error(double *y_est, double *y_target, uint16_t n_samples);

#endif // COST_HPP