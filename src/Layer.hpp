// Layer.hpp

#ifndef LAYER_HPP
#define LAYER_HPP

#include "Node.hpp"
#include <array>

template<size_t input_size, size_t layer_size>
struct DenseLayer {
    std::array<double, input_size> input;
    std::array<HiddenNode<input_size>, layer_size> nodes;
    std::array<double, layer_size> output;
    ActFunc layer_activation;

    void init_layer(ActFunc l_activation) {
        layer_activation = l_activation;
        for (size_t n_count = 0; n_count < layer_size; ++n_count) {
            nodes[n_count].input = input;
            for (size_t input_count = 0; input_count < input_size; ++input_count) {
                nodes[n_count].weights[input_count] = input_count + n_count * 10;
            }
            nodes[n_count].bias = 69 * (n_count % 2 - 1);
            nodes[n_count].activation = layer_activation;
        }
    }

    void forward() {
        for (size_t i = 0; i < layer_size; ++i) {
            output[i] = nodes[i].compute_output();
        }
    }
};

#endif // LAYER_HPP