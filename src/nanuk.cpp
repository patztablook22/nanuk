#include "nanuk.hpp"
using namespace nanuk;


Nanuk::Nanuk(const vector<unsigned>& topology) {
    init_network(topology);
}

void Nanuk::init_network(const vector<unsigned>& topology) {
    if (topology.size() < 2)
        throw invalid_argument(
            "there must be at least one layer other than the input layer"
        );

    layers.reserve(topology.size());
    
    // 1 input value per input neuron
    unsigned neurons_prev = 1;

    for (auto neurons: topology) {
        layers.push_back(
            Layer(neurons, neurons_prev)
        );
        neurons_prev = neurons;
    }
}

void Nanuk::inspect() const {
    for (int i = 0; i < layers.size(); i++) {
        cout << "layer " << i;
        cout << ": " << layers[i].size() << " neuron/s";
        cout << '\n';
    }
    cout << flush;
}

const vector<unsigned> Nanuk::topology() const {
    // return model's topology
    vector<unsigned> buff(layers.size());
    for (unsigned i = 0; i < buff.size(); i++)
        buff[i] = layers[i].size();
    return buff;
}

void Nanuk::feed_forward(const Tensor1D& input) {
    // feed input tensor to input neurons
    Layer& input_layer = layers[0];
    for (unsigned i = 0; i < input.size(); i++) {
        input_layer[i].feed_forward(input[i]);
    }
    
    // feed forward through hidden neurons
    for (unsigned i = 1; i < layers.size(); i++) {
        Layer& hidden_layer = layers[i];
        Layer& prev_layer   = layers[i - 1];
        
        for (Neuron& n: hidden_layer)
            n.feed_forward(prev_layer);
    }
}

Tensor1D Nanuk::output() const {
    const Layer& output_layer = layers.back();
    Tensor1D buff(output_layer.size());

    // collect output neuron memory into tensor
    transform(output_layer.begin(), output_layer.end(), buff.begin(),
        [&](auto& n) { return n.read(); }
    );
    return buff;
}

Tensor1D Nanuk::operator()(const Tensor1D& input) {
    if (input.size() != layers.front().size())
        throw("wrong number of features");
    feed_forward(input);
    return output();
}

void Nanuk::learning_params(size_t epochs, Scalar epsilon) {
    this->epochs   = epochs;
    this->epsilon  = epsilon;   
}

void Nanuk::learn(Tensor2D& features, Tensor2D& labels, Callback callback) {
    if (features.size() != labels.size())
        throw invalid_argument("features and labels are not of equal size");

    for (unsigned i = 0; i < epochs; i++) {
        Scalar cost = epoch(features, labels);
        if (callback != NULL)
            callback(i, cost);
    }
}

Scalar Nanuk::epoch(Tensor2D& features, Tensor2D& labels) {
    for (unsigned i = 0; i < features.size(); i++) {
        feed_forward(features[i]);
        propagate_back(labels[i]);
    }

    return cost_function(output(), labels.back());
}

void Nanuk::propagate_back(Tensor1D& labels) {
    {
        Layer& output_layer = layers.back();
        for (unsigned i = 0; i < output_layer.size(); i++)
            output_layer[i].calculate_gradient(labels[i]);
    }
    
    for (int i = layers.size() - 2; i >= 0; i--) {
        Layer& hidden_layer = layers[i];
        Layer& next_layer   = layers[i + 1];
        for (unsigned j = 0; j < hidden_layer.size(); j++)
            hidden_layer[j].calculate_gradient(next_layer, j);
    }
    
    for (unsigned i = 1; i < layers.size(); i++) {
        Layer& layer = layers[i];
        Layer& prev_layer = layers[i - 1];
        for (Neuron& n: layer)
            n.apply_gradient(epsilon, prev_layer);
    }
}