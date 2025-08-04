// Node.hpp

#ifndef NODE_HPP
#define NODE_HPP

#include "activations.hpp"
#include <cstddef>

template<size_t input_size>
struct HiddenNode {
    double input[input_size];
    double weight[input_size];
    double bias;
    double output;
    ActFunc activation;

    void compute_output() {
        int sum = 0;
        for (size_t i = 0; i < input_size; ++i) {
            sum += input[i] * weight[i];
        }
        output = activation(sum + bias);
    }
};

#endif // NODE_HPP