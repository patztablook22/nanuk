#include "nanuk.hpp"
#include "nanuk.cpp"
#include "neuron.cpp"

using namespace std;
using namespace nanuk;


int main(int argc, char** argv) {
    vector<unsigned> topology{1, 1};
    Nanuk n(topology);
    n.inspect();
    return 0;
}