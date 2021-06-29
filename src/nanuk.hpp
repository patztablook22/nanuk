#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
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
            void get_structure(Tensor1D&);
            void set_structure(const Tensor1D&);
            
    };
    

    class Nanuk {
        vector<Layer> layers;
        Scalar epsilon;

        public:
            Nanuk(const vector<unsigned>&);  // initialize manually
            Nanuk(ifstream&);                // import from a file
            void operator>>(ofstream&);      // export into a file
            void inspect();
            void learn(Tensor2D&, Tensor2D&);
            void learn(ifstream&, unsigned);
            Tensor1D operator()(Tensor1D&);
        
        private:
            void init_network(const vector<unsigned>&);

            Tensor1D output();
            void feed_forward(Tensor1D&);
            void propagate_back(Tensor1D&);
            void epoch(Tensor2D&, Tensor2D&);
    };

}