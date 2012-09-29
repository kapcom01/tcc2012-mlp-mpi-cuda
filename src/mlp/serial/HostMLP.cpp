#include "mlp/serial/HostMLP.h"

namespace ParallelMLP
{

//===========================================================================//

HostMLP::HostMLP(int mlpID)
	: MLP(mlpID)
{
	srand(time(NULL));
}

//===========================================================================//

HostMLP::HostMLP(string name, vector<uint> &units)
	: MLP(name, units)
{
	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
		addLayer(units[i], units[i + 1]);

	// Seta a saída e randomiza os pesos
	setOutput();
	randomize();
}

//===========================================================================//

HostMLP::~HostMLP()
{

}

//===========================================================================//

void HostMLP::addLayer(uint inUnits, uint outUnits)
{
	layers.push_back(new HostLayer(inUnits, outUnits));
}

//===========================================================================//

void HostMLP::train(HostExampleSet &training)
{
	// Inicializa a operação
	initOperation(training);

	// Épocas
	for (; epoch < training.getMaxEpochs(); epoch++)
	{
		cout << "Epoch " << epoch << endl;

		shuffleIndexes();
		totalError = 0;

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes[i];

			// Realiza o feedforward e salva os valores no conjunto
			feedforward(training.getInput(r));
			training.setOutput(r, output);

			// Calcula o erro e realiza o feedback
			calculateError(training.getTarget(r));
			feedback(training.getLearning());
		}

		totalError /= training.getSize() * training.getOutVars();

		// Condição de parada: erro menor do que um valor tolerado
		if (totalError < training.getTolerance())
			break;
	}

	// Finaliza a operação
	endOperation(training);
}

//===========================================================================//

void HostMLP::validate(HostExampleSet &validation)
{
	// Inicializa a operação
	initOperation(validation);

	// Para cada entrada
	for (uint i = 0; i < validation.getSize(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(validation.getInput(i));
		validation.setOutput(i, output);

		// Calcula o erro
		calculateError(validation.getTarget(i));
	}

	// Finaliza a operação
	endOperation(validation);
}

//===========================================================================//

void HostMLP::test(HostExampleSet &test)
{
	// Inicializa a operação
	initOperation(test);

	// Para cada entrada
	for (uint i = 0; i < test.getSize(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(test.getInput(i));
		test.setOutput(i, output);
	}

	// Finaliza a operação
	endOperation(test);
}

//===========================================================================//

}
