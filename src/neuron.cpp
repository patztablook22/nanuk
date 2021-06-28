#include "nanuk.hpp"

using namespace nanuk;

Neuron::Neuron(unsigned inputs)
    :synapsesIn(inputs)
{}

void Neuron::feed_forward(Scalar val) {
    memory = val;
}

void Neuron::feed_forward(Layer& prev_layer) {
    Scalar buff = bias;
    
    for (int i = 0; i < synapsesIn.size(); i++) {
        buff += synapsesIn[i].weight * prev_layer[i].memory;
    }
    
    memory = activation_function(buff);
}

Scalar Neuron::read() {
    return memory;
}