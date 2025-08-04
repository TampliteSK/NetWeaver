// activations.hpp

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

typedef double (*ActFunc)(double);

// The linear activation (aka do nothing)
#define LINEAR(x) (x)

double ReLU(double x);

#endif // ACTIVATIONS_HPP