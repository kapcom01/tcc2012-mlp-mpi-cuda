#include "arff/Driver.h"
#include "arff/Relation.h"

#include "mlp/serial/HostMLP.h"
#include "mlp/serial/HostExampleSet.h"
#include "mlp/cuda/DeviceMLP.h"
#include "mlp/cuda/DeviceExampleSet.h"

using namespace ParallelMLP;

void serialTrain(v_uint units, float learning, uint maxEpochs, float tolerance,
		const Relation &relation);

void cudaTrain(v_uint units, float learning, uint maxEpochs, float tolerance,
		const Relation &relation);

string program;

int main(int argc, char* argv[])
{
	try
	{
		program = argv[0];

		string usage = "Usage mode: " + program + " <serial|cuda|mpi> "
				"<neurons on input layer> [neurons on each hidden layer] "
				"<neurons on output layer> <arff file> <learning rate> "
				"<max epochs> <tolerance>";

		if (argc < 7)
			throw runtime_error(usage);

		string cmd = argv[1];
		string input = argv[argc - 4];
		float learning = atof(argv[argc - 3]);
		uint maxEpochs = atoi(argv[argc - 2]);
		float tolerance = atof(argv[argc - 1]);

		v_uint units;
		for (int i = 2; i < argc - 4; i++)
			units.push_back(atoi(argv[i]));

		Driver driver(input);
		const Relation &relation = driver.parse();

		cout << "Training MLP" << endl;

		if (cmd == "serial")
			serialTrain(units, learning, maxEpochs, tolerance, relation);

		else if (cmd == "cuda")
			cudaTrain(units, learning, maxEpochs, tolerance, relation);

		else
			throw runtime_error(usage);

		cout << "MLP trained" << endl;
	}
	catch (exception &ex)
	{
		cerr << ex.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void serialTrain(v_uint units, float learning, uint maxEpochs, float tolerance,
		const Relation &relation)
{
	HostExampleSet set(relation);
	set.setLearning(learning);
	set.setMaxEpochs(maxEpochs);
	set.setTolerance(tolerance);

	HostMLP mlp(units);
	mlp.train(set);
}

void cudaTrain(v_uint units, float learning, uint maxEpochs, float tolerance,
		const Relation &relation)
{
	DeviceExampleSet set(relation);
	set.setLearning(learning);
	set.setMaxEpochs(maxEpochs);
	set.setTolerance(tolerance);

	DeviceMLP mlp(units);
	mlp.train(set);
}
