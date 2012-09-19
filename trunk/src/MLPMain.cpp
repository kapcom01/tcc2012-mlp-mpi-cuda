#include "mlp/cuda/DeviceExampleSet.h"
#include "mlp/serial/BackpropMLP.h"
#include "database/ExampleSetAdapter.h"
#include "database/BackpropMLPAdapter.h"
#include <ctime>

using namespace ParallelMLP;

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
    	int relationID = 2;
    	int mlpID = 1;

//	    vector<uint> units = {4, 8, 3};
//	    BackpropMLP mlp("mlpiris", units);
//	    BackpropMLPAdapter::insert(mlp);

		DeviceExampleSet exampleSet(relationID, mlpID, TRAINING);
		ExampleSetAdapter::select(exampleSet);
		exampleSet.setProperties(0.4, 100000, 0.01);
		exampleSet.normalize();
		exampleSet.unnormalize();

//		BackpropMLP mlp(mlpID);
//		BackpropMLPAdapter::select(mlp);

//		mlp.train(exampleSet);

//		BackpropMLPAdapter::update(mlp, relationID);
//		ExampleSetAdapter::insert(exampleSet);
	}
	catch(exception &ex)
	{
		cerr << ex.what() << endl;
	}

    return EXIT_SUCCESS;
}
