#include "mlp/cuda/DeviceExampleSet.h"
#include "mlp/serial/HostMLP.h"
#include "database/ExampleSetAdapter.h"
#include "database/MLPAdapter.h"
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
    	int relationID = 1;
    	int mlpID = 2;

//	    vector<uint> units = {2, 3, 1};
//	    HostMLP mlp("mlpxor2", units);
//	    MLPAdapter::insert(mlp);

    	cout << "Reading example set" << endl;

		HostExampleSet exampleSet(relationID, mlpID, TRAINING);
		ExampleSetAdapter::select(exampleSet);
		exampleSet.setLearning(0.4);
		exampleSet.setMaxEpochs(10000);
		exampleSet.setTolerance(0.01);

		cout << "Example set read" << endl;
		cout << "Reading MLP" << endl;

		HostMLP mlp(mlpID);
		MLPAdapter::select(mlp);

		cout << "MLP read" << endl;
		cout << "Training MLP" << endl;

		mlp.train(exampleSet);

		cout << "MLP trained" << endl;

		MLPAdapter::update(mlp, relationID);
		ExampleSetAdapter::insert(exampleSet);
	}
	catch(exception &ex)
	{
		cerr << ex.what() << endl;
	}

    return EXIT_SUCCESS;
}
