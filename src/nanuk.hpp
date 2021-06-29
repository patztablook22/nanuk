#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>


namespace nanuk {
    
    using namespace std;
    using Scalar = double;
    using Tensor1D = vector<Scalar>;
    using Tensor2D = vector<Tensor1D>;
    
    class Neuron;
    using Layer  = vector<Neuron>;   // dense-only layer
    
    inline Scalar activation_function(const Scalar);
    inline Scalar activation_function_derivative(const Scalar);
    inline Scalar cost_function(const Tensor1D&, const Tensor1D&);
    inline Scalar cost_function_derivative(const Tensor1D&, const Tensor1D&);


    class Neuron {
        vector<Scalar> dendrites;
        Scalar bias;
        Scalar sum, activation, gradient;

        public:
            Neuron(unsigned);
            Scalar read();
            void feed_forward(Scalar);
            void feed_forward(Layer&);
            void calculate_gradient(Scalar);
            void calculate_gradient(Layer&, unsigned);
            void apply_gradient(Scalar, Layer&);
    };
    

    class Nanuk {
        vector<Layer> layers;
        Scalar epsilon;

        public:
            Nanuk(vector<unsigned>&);
            Nanuk(ifstream);
            void learn(Tensor2D&, Tensor2D&);
            Tensor1D operator()(Tensor1D&);
            void inspect();
        
        private:
            Tensor1D output();
            void feed_forward(Tensor1D&);
            void propagate_back(Tensor1D&);
            void epoch(Tensor2D&, Tensor2D&);
    };

}