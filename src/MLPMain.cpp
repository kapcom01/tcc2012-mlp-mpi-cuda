#include "mlp/BackpropMLP.h"
#include "database/ExampleSetAdapter.h"
#include "database/BackpropMLPAdapter.h"
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

    try
	{
//	    vuint units = {4, 8, 3};
//	    BackpropMLP mlp("mlpiris", units);
//	    BackpropMLPAdapter::insert(mlp);

		ExampleSet exampleSet(3, TRAINING);
		ExampleSetAdapter::select(exampleSet, 1);
		exampleSet.learning = 0.4;
		exampleSet.maxEpochs = 100000;
		exampleSet.tolerance = 0.01;

		BackpropMLP mlp(1);
		BackpropMLPAdapter::select(mlp);

		mlp.train(exampleSet);

		BackpropMLPAdapter::update(mlp, 3);
	}
	catch(exception &ex)
	{
		cerr << ex.what() << endl;
	}

    return EXIT_SUCCESS;
}
