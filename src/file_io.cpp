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
        
        if (i == 0) continue; // no data for input layer required
        
        Tensor1D structure(layers[i - 1].size() + 1);
        
        // load neurons
        for (Neuron& hidden_neuron: layer) {
            getline(is, buff);
            stringstream line(buff);
            for (Scalar& param: structure)
                line >> param;
            hidden_neuron.set_structure(structure);
        }        
    }
}


void Nanuk::operator>>(ofstream& os) const {
    // header
    os << "nanuk model" << '\n';

    // save topology
    os << "topology";
    for (auto& layer: layers)
        os << ' ' << layer.size();
    os << '\n';
    
    // save layers
    for (unsigned i = 0; i < layers.size(); i++) {
        auto layer = layers[i];
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


void Nanuk::learn(ifstream& csv, Callback callback, unsigned header_lines) {
    string buff;
    
    // features first, labels second
    Tensor2D features, labels;
    vector<Tensor2D*> datasets{ &features, &labels };
    
    // ignore header lines
    for (unsigned i = 0; i < header_lines; i++)
        getline(csv, buff);
    

    // parse data lines
    while (csv) {
        getline(csv, buff);

        // skip empty lines
        if (buff.empty()) continue;

        stringstream line(buff);
        Scalar val;
        
        // add record to features and labels tensors
        
        features.push_back(
            Tensor1D(layers.front().size())
        );

        labels.push_back(
            Tensor1D(layers.back().size())
        );
        
        // read values into datasets
        for (Tensor2D* dataset: datasets) {
            for (Scalar& data: dataset->back()) {
                getline(line, buff, ',');
                stringstream val(buff);
                val >> data;
            }
        }
    }

    /*
    // printing parsed values for debug
    for (unsigned i = 0; i < features.size(); i++) {
        for (Tensor2D* dataset: datasets) {
            Tensor1D row = (*dataset)[i];
            for (Scalar data: row)
                cout << data << '\t';
        }
        cout << '\n';
    }
    */
    
    learn(features, labels, callback);
}