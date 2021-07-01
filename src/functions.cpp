#include "nanuk.hpp"
using namespace nanuk;


Scalar nanuk::activation_function(const Scalar x) {
    return ::tanh(x);
}

Scalar nanuk::activation_function_derivative(const Scalar x) {
    // tanh derivative approximation
    // return 1 - x*x;
    // tanh derivative = sech^2(x) = 1/cosh^2(x)
    Scalar tmp = ::cosh(x);
    return 1 / (tmp * tmp);
}

Scalar nanuk::cost_function(const Tensor1D& t1, const Tensor1D& t2) {    
    /*
     *  sum( (t1-t2)^2 )/n
     */

    Scalar buff = 0;

    for (unsigned i = 0; i < t1.size(); i++) {
        Scalar tmp = t1[i] - t2[i];
        buff += tmp*tmp;
    }
    buff /= t1.size();
    return buff;
}

Scalar nanuk::cost_function_derivative(const Tensor1D& t1, const Tensor1D& t2) {
    /*
     *  sum( -2 * (t1-t2) )/n
     */
    
    Scalar buff = 0;

    for (unsigned i = 0; i < t1.size(); i++) {
        Scalar tmp = t1[i] - t2[i];
        buff -= 2 * tmp;
    }
    buff /= t1.size();
    return buff;
}