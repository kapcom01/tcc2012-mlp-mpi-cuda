#include "arff/Driver.h"
#include "arff/Relation.h"

#include "database/RelationAdapter.h"
#include "database/ExampleSetAdapter.h"
#include "database/MLPAdapter.h"

#include "mlp/serial/HostMLP.h"
#include "mlp/cuda/DeviceMLP.h"

using namespace ParallelMLP;

#define PARSE_ARFF	0
#define CREATE_MLP	1
#define TRAIN_MLP	2

void parseARFF(int argc, char* argv[]);
void createMLP(int argc, char* argv[]);
void trainMLP(int argc, char* argv[]);
void fastTrain(int argc, char* argv[]);

string program;

int main(int argc, char* argv[])
{
	try
	{
		program = argv[0];

		string usage = "Usage mode: " + program + " <parse_arff|create_mlp|"
				"train_mlp> [options]";

		if (argc < 2)
			throw runtime_error(usage);

		string cmd = argv[1];

		if (cmd == "parse_arff")
			parseARFF(argc, argv);

		else if (cmd == "create_mlp")
			createMLP(argc, argv);

		else if (cmd == "train_mlp")
			trainMLP(argc, argv);

		else if (cmd == "fast_train")
			fastTrain(argc, argv);

		else
			throw runtime_error(usage);
	}
	catch (exception &ex)
	{
		cerr << ex.what() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void parseARFF(int argc, char* argv[])
{
	string usage = "Usage mode: " + program + " parse_arff <arff file>";

	if (argc != 3)
		throw runtime_error(usage);

	string input = argv[2];
	Driver driver(input);

	Relation* relation = driver.parse();
	RelationAdapter::insert(*relation);
}

void createMLP(int argc, char* argv[])
{
	string usage = "Usage mode: " + program + " create_mlp <mlp name> "
			"<neurons on input layer> [neurons on each hidden layer] "
			"<neurons on output layer>";

	if (argc < 5)
		throw runtime_error(usage);

	string name = argv[2];
	v_uint units;

	for (int i = 3; i < argc; i++)
		units.push_back(atoi(argv[i]));

	HostMLP mlp(name, units);
	MLPAdapter::insert(mlp);
}

void trainMLP(int argc, char* argv[])
{
	string usage = "Usage mode: " + program + " train_mlp <serial|cuda|mpi> "
			"<mlp id> <relation id> <learning rate> <max epochs> <tolerance>";

	if (argc != 8)
		throw runtime_error(usage);

//	string mode = argv[2];
//	uint mlpID = atoi(argv[3]);
//	uint relationID = atoi(argv[4]);
//	float learning = atof(argv[5]);
//	uint maxEpochs = atoi(argv[6]);
//	float tolerance = atof(argv[7]);
//
//	ExampleSet* exampleSet;
//	MLP* mlp;
//
//	if (mode == "serial")
//	{
//		exampleSet = new HostExampleSet(relationID, mlpID, TRAINING);
//		mlp = new HostMLP(mlpID);
//	}
//	else if (mode == "cuda")
//	{
//		exampleSet = new DeviceExampleSet(relationID, mlpID, TRAINING);
//		mlp = new DeviceMLP(mlpID);
//	}
//	else
//		throw runtime_error(usage);
//
//	cout << "Reading example set" << endl;
//
//	ExampleSetAdapter::select(*exampleSet);
//	exampleSet->setLearning(learning);
//	exampleSet->setMaxEpochs(maxEpochs);
//	exampleSet->setTolerance(tolerance);
//
//	cout << "Example set read" << endl;
//	cout << "Reading MLP" << endl;
//
//	MLPAdapter::select(*mlp);
//
//	cout << "MLP read" << endl;
//	cout << "Training MLP" << endl;
//
//	if (mode == "serial")
//	{
//		HostMLP* hmlp = (HostMLP*) mlp;
//		HostExampleSet* hset = (HostExampleSet*) exampleSet;
//		hmlp->train(hset);
//	}
//	else if (mode == "cuda")
//	{
//		DeviceMLP* dmlp = (DeviceMLP*) mlp;
//		DeviceExampleSet* dset = (DeviceExampleSet*) exampleSet;
//		dmlp->train(dset);
//	}
//
//	cout << "MLP trained" << endl;
//
////		MLPAdapter::update(*mlp, relationID);
////		ExampleSetAdapter::insert(*exampleSet);
//
//	delete exampleSet;
//	delete mlp;
}

void transform(Relation &relation, ExampleSet &set)
{
	set.setSize(relation.getNInstances());

	for (uint i = 0; i < relation.getNInstances(); i++)
	{
		const Instance &ins = relation.getInstance(i);

		for (uint j = 0; j < relation.getNAttributes(); j++)
		{
			bool isTarget = (j + 1 == relation.getNAttributes());

			if (isTarget)
				set.addBias();

			if (relation.getAttribute(j).getType() == NUMERIC)
				set.addValue(ins[j]->getNumber(), isTarget);
			else
				set.addValue(ins[j]->getNominal(), relation.getAttribute(j).getNominalCard(), isTarget);
		}
	}

	for (uint j = 0; j < relation.getNAttributes(); j++)
	{
		bool isTarget = (j + 1 == relation.getNAttributes());

		if (relation.getAttribute(j).getType() == NUMERIC)
		{
			float min = 1000000, max = -1000000;

			for (uint i = 0; i < relation.getNInstances(); i++)
			{
				const Instance &ins = relation.getInstance(i);

				if (ins[j]->getNumber() > max)
					max = ins[j]->getNumber();
				if (ins[j]->getNumber() < min)
					min = ins[j]->getNumber();
			}

			set.addStat(min, max, -1, 1, isTarget);
		}
		else
			set.addStat(-1, 1, relation.getAttribute(j).getNominalCard(), isTarget);
	}

	set.done();

}

void fastTrain(int argc, char* argv[])
{
	string usage = "Usage mode: " + program + " train_mlp <serial|cuda|mpi> "
			"<neurons on input layer> [neurons on each hidden layer] "
			"<neurons on output layer> <arff file> <learning rate> "
			"<max epochs> <tolerance>";

	if (argc < 8)
		throw runtime_error(usage);

	string mode = argv[2];
	string input = argv[argc - 4];
	float learning = atof(argv[argc - 3]);
	uint maxEpochs = atoi(argv[argc - 2]);
	float tolerance = atof(argv[argc - 1]);
	v_uint units;

	for (int i = 3; i < argc - 4; i++)
		units.push_back(atoi(argv[i]));

	Driver driver(input);
	Relation* relation = driver.parse();

	cout << "Training MLP" << endl;

	if (mode == "serial")
	{
		HostExampleSet set;
		HostMLP mlp("fast", units);

		transform(*relation, set);
		set.setLearning(learning);
		set.setMaxEpochs(maxEpochs);
		set.setTolerance(tolerance);

		mlp.train(&set);
	}
	else if (mode == "cuda")
	{
		DeviceExampleSet set;
		DeviceMLP mlp(units);

		transform(*relation, set);
		set.setLearning(learning);
		set.setMaxEpochs(maxEpochs);
		set.setTolerance(tolerance);

		mlp.train(set);
	}
	else
		throw runtime_error(usage);


	cout << "MLP trained" << endl;
}
