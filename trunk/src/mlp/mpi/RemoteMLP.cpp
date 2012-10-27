#include "mlp/mpi/RemoteMLP.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteMLP::RemoteMLP(v_uint &units)
	: MLP(units)
{
	hid = COMM_WORLD.Get_rank();
	hosts = COMM_WORLD.Get_size();

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i + 1 == units.size() - 1)
			layers.push_back(new RemoteOutLayer(units[i], units[i + 1],
					hid, hosts));
		else
			layers.push_back(new RemoteLayer(units[i], units[i + 1],
					hid, hosts));
	}

	// Seta os ponteiros para a primeira e última camada
	linkLayers();
}

//===========================================================================//

RemoteMLP::~RemoteMLP()
{

}

//===========================================================================//

void RemoteMLP::train(ExampleSet &training)
{
	// Inicializa a operação
	initOperation(training);

	// Épocas
	for (epoch = 0; epoch < training.getMaxEpochs(); epoch++)
	{
		if (hid == 0)
			cout << "Epoch " << epoch << endl;

		indexes.randomize();
		outLayer->clearError();

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes.get(i);

			if (hid == 0)
				cout << "Input " << r << endl;

			// Realiza o feedforward e salva os valores no conjunto
			feedforward(training.getInput(r));
			training.setOutput(r, outLayer->getFuncSignal());

			// Realiza o feedback
			feedbackward(training.getTarget(r), training.getLearning());
		}

		// Condição de parada: erro menor do que um valor tolerado
		if (outLayer->getError() < training.getTolerance())
			break;
	}

	// Finaliza a operação
	endOperation(training);
}

//===========================================================================//

}
