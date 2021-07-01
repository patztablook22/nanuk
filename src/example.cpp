#include "nanuk.hpp"

using namespace std;
using namespace nanuk;

void hr() { cout << string(32, '=') << '\n'; }
void test(Nanuk& n) {
     for (Scalar i = 0; i <= 1; i += .2)
        cout << i << "\t => " << n({i}) [0] << endl;
}


int main(int argc, char** argv) {
    ifstream training("training.csv");
    ofstream model("MyModel.nanuk");

    Nanuk n({1, 1});
    
    n.learning_params(
    // epochs:
        1000,
    // epsilon:
        .1
    );

    hr(); n.learn(training); hr();
    
    test(n);
    n >> model;
    return EXIT_SUCCESS;
}