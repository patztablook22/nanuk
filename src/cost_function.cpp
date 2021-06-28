#include "nanuk.hpp"
#include <cmath>
using namespace nanuk;

// Root Mean Square, tensors of equal size assumed
Scalar cost_function(Tensor1D& t1, Tensor1D& t2) {    
    Scalar buff = 0;

    for (unsigned i = 0; i < t1.size(); i++) {
        Scalar tmp = t1[i] - t2[i];
        buff += tmp*tmp;
    }
    buff /= t1.size();
    return sqrt(buff);
}