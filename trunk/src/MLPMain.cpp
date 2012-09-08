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
    	int relationID = 4;
    	int mlpID = 2;

//	    vuint units = {4, 8, 3};
//	    BackpropMLP mlp("mlpiris", units);
//	    BackpropMLPAdapter::insert(mlp);

		ExampleSet exampleSet(relationID, mlpID, TRAINING);
		ExampleSetAdapter::select(exampleSet);
		exampleSet.learning = 0.4;
		exampleSet.maxEpochs = 100000;
		exampleSet.tolerance = 0.01;

		BackpropMLP mlp(mlpID);
		BackpropMLPAdapter::select(mlp);

		mlp.train(exampleSet);

		BackpropMLPAdapter::update(mlp, relationID);
		ExampleSetAdapter::insert(exampleSet);
	}
	catch(exception &ex)
	{
		cerr << ex.what() << endl;
	}

    return EXIT_SUCCESS;
}
