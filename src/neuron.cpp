#include "nanuk.hpp"
#define NEURON_INIT -1
using namespace nanuk;


Scalar Neuron::read() const {
    return activation;
}

Neuron::Neuron(unsigned inputs)
    :dendrites(inputs, NEURON_INIT)
    ,bias(NEURON_INIT)
{}

void Neuron::feed_forward(Scalar val) {
    activation = val;
}

void Neuron::feed_forward(Layer& prev_layer) {
    sum = bias;
    
    for (int i = 0; i < dendrites.size(); i++)
        sum += dendrites[i] * prev_layer[i].activation;

    activation = activation_function(sum);
}

void Neuron::calculate_gradient(Scalar label) {
    gradient  = -activation_function_derivative(sum);
    gradient *= cost_function_derivative(
        {activation}, {label}
    );
}

void Neuron::calculate_gradient(Layer& next_layer, unsigned index) {
    gradient  = 0;
    // weighted chain to calculated cost_function_derivative above
    for (unsigned i = 0; i < next_layer.size(); i++)
        gradient += next_layer[i].gradient * next_layer[i].dendrites[index];
    gradient *= activation_function_derivative(sum);
}

void Neuron::apply_gradient(Scalar epsilon, Layer& prev_layer) {
    // update dendrites
    for (unsigned i = 0; i < dendrites.size(); i++) {
        Scalar& d = dendrites[i];
        Scalar  a = prev_layer[i].activation;
        d -= epsilon * gradient * a;
    }

    // update bias
    bias -= epsilon * gradient; // * a = 1
}

void Neuron::get_structure(Tensor1D& t) const {
    t = dendrites;
    t.push_back(bias);
}

void Neuron::set_structure(const Tensor1D& t) {
    copy(
        t.begin(), t.end() - 1,
        dendrites.begin()
    );
    bias = t.back();
}