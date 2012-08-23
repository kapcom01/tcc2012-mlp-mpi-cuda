#include "mlp/BackpropMLP.h"
#include "database/ExampleSetAdapter.h"
#include <ctime>

using namespace MLP;
using namespace Database;

int main(int argc, char* argv[])
{
    if(argc != 1)
    {
        cerr << "Usage mode: " << argv[0] << endl;
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    ExampleSet exampleSet;
    ExampleSetAdapter::select(2, exampleSet);
    exampleSet.learningRate = 0.1;
    exampleSet.momentum = 0.9;
    exampleSet.maxEpochs = 100000;
    exampleSet.maxTolerance = 0.01;
    exampleSet.minSuccessRate = 0.95;

    vector<uint> units = {2, 3, 1};
    BackpropMLP mlp(units, LOGISTIC);

    mlp.train(exampleSet);

    return EXIT_SUCCESS;
}
