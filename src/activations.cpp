// activations.cpp

#include "activations.hpp"

double ReLU(double x) {
    if (x <= 0) return 0;
    return x;
}