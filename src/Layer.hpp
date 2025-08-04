// Layer.hpp

#ifndef LAYER_HPP
#define LAYER_HPP

template<size_t input_size, size_t output_size>
struct DenseLayer {
    double input[input_size];
    double weight[input_size];
    double bias;
};

#endif // LAYER_HPP