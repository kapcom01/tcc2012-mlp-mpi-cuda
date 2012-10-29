#include "arff/Driver.h"
#include "arff/Relation.h"

#include "mlp/mpi/RemoteMLP.h"
#include "mlp/mpi/RemoteExampleSet.h"

using namespace ParallelMLP;

int main(int argc, char* argv[])
{
	try
	{
		Init(argc, argv);

		string program = argv[0];
		int hid = COMM_WORLD.Get_rank();

		string usage = "Usage mode: " + program + " <neurons on input layer> "
				"[neurons on each hidden layer] <neurons on output layer> "
				"<arff file> <learning rate> <max epochs> <tolerance>";

		if (argc < 6)
			throw runtime_error(usage);

		string input = argv[argc - 4];
		float learning = atof(argv[argc - 3]);
		uint maxEpochs = atoi(argv[argc - 2]);
		float tolerance = atof(argv[argc - 1]);

		v_uint units;
		for (int i = 1; i < argc - 4; i++)
			units.push_back(atoi(argv[i]));

		if (hid == 0)
			cout << "Parsing file..." << endl;

		Driver driver(input);
		const Relation &relation = driver.parse();

		if (hid == 0)
			cout << "Training MLP..." << endl;

		RemoteExampleSet set(relation);
		set.setLearning(learning);
		set.setMaxEpochs(maxEpochs);
		set.setTolerance(tolerance);

		RemoteMLP mlp(units);
		mlp.train(set);

		if (hid == 0)
		{
			cout << "Done! Results:" << endl;
			cout << " |-> Error: " << set.getError() << endl;
			cout << " |-> Time: " << set.getTime() << endl;
			cout << " |-> Epochs: " << set.getEpochs() << endl;
		}

		Finalize();
	}
	catch (exception &ex)
	{
		cerr << ex.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
