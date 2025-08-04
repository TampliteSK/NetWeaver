// Node.hpp

#ifndef NODE_HPP
#define NODE_HPP

#include "activations.hpp"
#include <cstddef>
#include <array>

template<size_t input_size>
struct HiddenNode {
    std::array<double, input_size> input;
    std::array<double, input_size> weights;
    double bias;
    double output;
    ActFunc activation;

    double compute_output() {
        int sum = 0;
        for (size_t i = 0; i < input_size; ++i) {
            sum += input[i] * weights[i];
        }
        output = activation(sum + bias);
        return output;
    }
};

#endif // NODE_HPP