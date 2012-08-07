#include "mlp/MLP_BP.h"
#include <ctime>

using namespace MLP;

int main(int argc, char* argv[])
{
    if(argc != 1)
    {
        cerr << "Usage mode: " << argv[0] << endl;
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    InputSet inputSet(4, 2, 1);
    inputSet.input[0][0] = 0, inputSet.input[0][1] = 0, inputSet.expectedOutput[0][0] = 0;
    inputSet.input[1][0] = 0, inputSet.input[1][1] = 1, inputSet.expectedOutput[1][0] = 1;
    inputSet.input[2][0] = 1, inputSet.input[2][1] = 0, inputSet.expectedOutput[2][0] = 1;
    inputSet.input[3][0] = 1, inputSet.input[3][1] = 1, inputSet.expectedOutput[3][0] = 0;
    inputSet.learningRate = 0.5, inputSet.searchTime = 200;
    inputSet.maxIterations = 10000;
    inputSet.maxTolerance = 0.01;
    inputSet.minSuccessRate = 0.95;

    uint units[] = {2, 3, 1};
    MLP_BP mlp(2, units, LOGISTIC, CLASSIFICATION);
    mlp.train(&inputSet);

    return EXIT_SUCCESS;
}
