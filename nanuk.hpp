#pragma once

#include <vector>
#include <valarray>
#include <boost/numeric/ublas/matrix.hpp>
#include <fstream>

namespace nanuk {
    
    using namespace std;
    using boost::numeric::ublas::matrix;
    using Scalar = double;
    using Matrix = matrix<Scalar>;
    using Layer  = vector<Neuron>;

    struct Synapse {
        Scalar weight;
    };
    
    class Neuron {
        Scalar bias;
        vector<Synapse> synIn;
    
        public:
            Scalar operator()(Layer&); // feed forward

    };

    class Nanuk {
        vector<Layer> layers;

        public:
            Nanuk();
            void operator<<(vector<vector<Matrix>>&); // future fit/train function
            void operator()(vector<Matrix>&); // future execution function
            void inspect();

    };
}