# nanuk

C++ **N**eur**a**l **N**etw**u**r**k**

Machine Learning from scratch.

_NOTE: `nanuk-ruby` API and `nanuk-on-rails` integration planned_

# C++ demo

```C++
using namespace nanuk;

Nanuk n({4, 3, 2, 1}); // 4 input neurons, 2 hidden layers, 1 output neuron
n.learing_params(
    10000,  // epochs
    .4      // learning rate [epsilon]
);

// train model
std::ifstream data("data.csv");
n.learn(data);

// apply model
Tensor1D output = n({ /* input vector */ });
std::cout << output[0] << std::endl;
```

# Dependencies
- `C++14`
- `CMake`

# Setup
run `./setup.sh` to generate `libnanuk.so` shared object and `example` executable
