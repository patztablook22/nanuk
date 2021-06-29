#include "nanuk.hpp"
using namespace nanuk;


inline Scalar nanuk::activation_function(const Scalar x) {
    return ::tanh(x);
}

inline Scalar nanuk::activation_function_derivative(const Scalar x) {
    // tanh derivative approximation
    return 1 - x*x;
}

inline Scalar nanuk::cost_function(const Tensor1D& t1, const Tensor1D& t2) {    
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

inline Scalar nanuk::cost_function_derivative(const Tensor1D& t1, const Tensor1D& t2) {
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