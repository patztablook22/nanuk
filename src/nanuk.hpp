#pragma once

#include <vector>
#include <fstream>
#include <cmath>

namespace nanuk {
    
    using namespace std;
    using Scalar = double;
    using Tensor1D = vector<Scalar>;
    using Tensor2D = vector<Tensor1D>;
    
    class Neuron;
    using Layer  = vector<Neuron>;   // dense-only layer
    
    Scalar (* activation_function)(Scalar) = &::tanh;
    Scalar    cost_function(Tensor1D&, Tensor1D&);


    struct Synapse {
        Scalar weight;
    };
    

    class Neuron {
        Scalar bias;
        vector<Synapse> synapsesIn;
        Scalar memory;

        public:
            Neuron(unsigned);
            void feed_forward(Scalar);
            void feed_forward(Layer&);
            Scalar read();
    };
    

    class Nanuk {
        vector<Layer> layers;

        public:
            Nanuk(vector<unsigned>&);
            void operator<<(Tensor2D&);
            Tensor1D operator()(Tensor1D&);
            void inspect();
        
        private:
            Tensor1D output();
            void feed_forward(Tensor1D&);
            void propagate_back(Tensor1D&);
            
    };
}