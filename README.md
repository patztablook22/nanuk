# nanuk

A **N**eur**a**l **N**etw**u**r**k**

project to learn C++, various libraries, CMake, and Machine Learning from scratch

Ruby API is planned

# C++ demo

```C++
using namespace nanuk;

Nanuk n({4, 3, 2, 1}); // 4 input neurons, 2 hidden layeres, 1 output neuron
n.learing_params(
    10000,  // epochs
    .4      // learning rate [epsilon]
);

// train model
std::ifstream data("data.csv");
n.learn(data);

// apply model
Scalar output = n({ /* input vector */});
std::cout << output[0] << std::endl;
```
