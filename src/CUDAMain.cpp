#include "arff/Driver.h"
#include "arff/Relation.h"

#include "mlp/cuda/DeviceMLP.h"
#include "mlp/cuda/DeviceExampleSet.h"

using namespace ParallelMLP;

int main(int argc, char* argv[])
{
	try
	{
		string program = argv[0];

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

		cout << "Parsing file..." << endl;

		Driver driver(input);
		const Relation &relation = driver.parse();

		cout << "Training MLP..." << endl;

		DeviceExampleSet set(relation);
		set.setLearning(learning);
		set.setMaxEpochs(maxEpochs);
		set.setTolerance(tolerance);

		DeviceMLP mlp(units);
		mlp.train(set);

		cout << "Done! Results:" << endl;
		cout << " |-> Error: " << set.getError() << endl;
		cout << " |-> Time: " << set.getTime() << endl;
		cout << " |-> Epochs: " << set.getEpochs() << endl;
	}
	catch (exception &ex)
	{
		cerr << ex.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
