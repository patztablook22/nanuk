#include "nanuk.hpp"
using namespace nanuk;


Nanuk::Nanuk(ifstream& is) {
    string buff;
    
    // header check
    getline(is, buff);
    if (buff != "nanuk model")
        throw invalid_argument("not a nanuk model");
    

    // load topology
    is >> buff; // initial "topology" keyword
    vector<unsigned> topology;
    {
        unsigned neurons;
        getline(is, buff);
        stringstream line(buff);
        while (line >> neurons)
            topology.push_back(neurons);
    } 
    init_network(topology);
    

    // load layers
    for (unsigned i = 0; i < layers.size(); i++) {
        Layer& layer = layers[i];
        
        getline(is, buff); // empty line
        getline(is, buff); // "layer i:"
        cout << "|" << buff << "|" << endl;
        
        if (i == 0) continue; // no data for input layer required
        
        Tensor1D structure(layers[i - 1].size() + 1);
        
        // load neurons
        for (Neuron& hidden_neuron: layer) {
            getline(is, buff);
            stringstream line(buff);
            for (Scalar& param: structure)
                line >> param;
            for (Scalar param: structure)
                cout << param << " ";
            cout << endl;
            hidden_neuron.set_structure(structure);
        }        
    }
}


void Nanuk::operator>>(ofstream& os) {
    // header
    os << "nanuk model" << '\n';

    // save topology
    os << "topology";
    for (Layer& layer: layers)
        os << ' ' << layer.size();
    os << '\n';
    
    // save layers
    for (unsigned i = 0; i < layers.size(); i++) {
        Layer& layer = layers[i];
        os << "\n";
        os << "layer " << i << ":\n";
        if (i == 0) continue;
        
        // save neurons
        for (Neuron& n: layer) {
            Tensor1D structure;
            n.get_structure(structure);
            for (Scalar x: structure)
                os << x << ' ';
            os << '\n';
        }
    }
}