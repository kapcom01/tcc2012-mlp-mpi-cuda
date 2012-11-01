#include "arff/Driver.h"
#include "arff/Relation.h"

#include "mlp/mpi/RemoteMLP.h"
#include "mlp/mpi/RemoteExampleSet.h"

using namespace ParallelMLP;

void fastTrain(int argc, char* argv[]);
void normalTrain(int argc, char* argv[]);
void train(ExampleSet &set, v_uint units, float learning, uint maxEpochs,
		float tolerance);

string program;
uint hid;

int main(int argc, char* argv[])
{
	Init(argc, argv);

	try
	{
		hid = COMM_WORLD.Get_rank();

		program = argv[0];
		string usage = "Usage mode: " + program + " <fast|normal>";

		if (argc < 2)
			throw runtime_error(usage);

		string mode = argv[1];

		if (mode == "fast")
			fastTrain(argc, argv);
		else if (mode == "normal")
			normalTrain(argc, argv);
		else
			throw runtime_error(usage);
	}
	catch (exception &ex)
	{
		if (hid == 0)
			cerr << ex.what() << endl;
		return EXIT_FAILURE;
	}

	Finalize();

	return EXIT_SUCCESS;
}

void fastTrain(int argc, char* argv[])
{
	string usage = "Usage mode: " + program + " fast "
			"<neurons on input layer> [neurons on each hidden layer] "
			"<neurons on output layer> <number of instances> ";

	if (argc < 5)
		throw runtime_error(usage);

	uint size = atoi(argv[argc - 1]);

	v_uint units;
	for (int i = 2; i < argc - 1; i++)
		units.push_back(atoi(argv[i]));

	RemoteExampleSet set(size, units.front(), units.back());
	train(set, units, 0.5, 100, 0.01);
}

void normalTrain(int argc, char* argv[])
{
	string usage = "Usage mode: " + program + " normal "
			"<neurons on input layer> [neurons on each hidden layer] "
			"<neurons on output layer> <arff file> <learning rate> "
			"<max epochs> <tolerance>";

	if (argc < 8)
		throw runtime_error(usage);

	string input = argv[argc - 4];
	float learning = atof(argv[argc - 3]);
	uint maxEpochs = atoi(argv[argc - 2]);
	float tolerance = atof(argv[argc - 1]);

	v_uint units;
	for (int i = 2; i < argc - 4; i++)
		units.push_back(atoi(argv[i]));

	Driver driver(input);
	const Relation &relation = driver.parse();

	RemoteExampleSet set(relation);
	train(set, units, learning, maxEpochs, tolerance);
}

void train(ExampleSet &set, v_uint units, float learning, uint maxEpochs,
		float tolerance)
{
	if (hid == 0)
		cout << "Training MLP..." << endl;

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
}

