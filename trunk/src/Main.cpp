#include "mlp/MLPerceptron.h"
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

    Settings settings(2);
    settings.activationType = LOGISTIC;
    settings.initialLR = 0.9, settings.minLR = 0.01, settings.maxLR = 0.9;
    settings.maxTolerance = 0.01;
    settings.minSuccessRate = 0.95;
    settings.units[0] = 2, settings.units[1] = 3, settings.units[2] = 1;

    InputSet inputSet(4, 2, 1);
    inputSet.input[0][0] = 0, inputSet.input[0][1] = 0, inputSet.expectedOutput[0][0] = 0;
    inputSet.input[1][0] = 0, inputSet.input[1][1] = 1, inputSet.expectedOutput[1][0] = 1;
    inputSet.input[2][0] = 1, inputSet.input[2][1] = 0, inputSet.expectedOutput[2][0] = 1;
    inputSet.input[3][0] = 1, inputSet.input[3][1] = 1, inputSet.expectedOutput[3][0] = 0;

    MLPerceptron mlp(&settings);
    mlp.train(&inputSet);

    return EXIT_SUCCESS;
}
