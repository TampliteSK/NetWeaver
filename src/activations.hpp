// activations.hpp

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <functional>

using ActFunc = std::function<double(double)>;

// The linear activation (aka do nothing)
#define LINEAR(x) (x)

double ReLU(double x);

#endif // ACTIVATIONS_HPP