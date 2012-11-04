#include "arff/Driver.h"
#include "arff/Relation.h"

#include "mlp/cuda/DeviceMLP.h"
#include "mlp/cuda/DeviceExampleSet.h"

using namespace ParallelMLP;

void fastTrain(int argc, char* argv[]);
void normalTrain(int argc, char* argv[]);
void train(ExampleSet &set, v_uint units, float learning, uint maxEpochs,
		float tolerance);

string program;

int main(int argc, char* argv[])
{
	cublasCreate(&DeviceUtil::cublas);
	cublasSetPointerMode(DeviceUtil::cublas, CUBLAS_POINTER_MODE_DEVICE);

	try
	{
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
		cerr << ex.what() << endl;
		return EXIT_FAILURE;
	}

	cublasDestroy(DeviceUtil::cublas);

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

	DeviceExampleSet set(size, units.front(), units.back());
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

	DeviceExampleSet set(relation);
	train(set, units, learning, maxEpochs, tolerance);
}

void train(ExampleSet &set, v_uint units, float learning, uint maxEpochs,
		float tolerance)
{
	cout << "Training MLP..." << endl;

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

