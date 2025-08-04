// NetWeaver.cpp

#include "Layer.hpp"
#include <iostream>

int main(void) {
    DenseLayer<2, 2> layer;
    layer.input[0] = -69;
    layer.input[1] = 420;
    layer.init_layer(ReLU);
    layer.forward();

    std::cout << "Output: [";
    for (int i = 0; i < 2; ++i) {
        std::cout << " " << layer.output[i] << ", ";
    }
    std::cout << "]\n";
}