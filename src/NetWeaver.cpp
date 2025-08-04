// NetWeaver.cpp

#include "Node.hpp"
#include <iostream>

int main(void) {
    HiddenNode<2> node;
    node.input[0] = 69;
    node.input[1] = 420;
    node.weight[0] = -3;
    node.weight[1] = 2;
    node.bias = -1000;
    node.activation = ReLU;

    node.compute_output();
    std::cout << "Output: " << node.output << "\n";
}