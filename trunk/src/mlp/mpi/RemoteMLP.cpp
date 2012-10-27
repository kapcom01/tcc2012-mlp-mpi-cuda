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
		cout << hid << ": Epoch " << epoch << endl;

		indexes.randomize();
		outLayer->clearError();

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes.get(i);

			cout << hid << ": Input " << r << endl;

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

void RemoteMLP::feedforward(const float* input)
{
	// Propaga a entrada para a primeira camada escondida
	inLayer->feedforward(input);

	cout << hid << ": FF Layer 1" << endl;

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
	{
		layers[i]->feedforward(layers[i - 1]->getFuncSignal());
		cout << hid << ": FF Layer " << (i+1) << endl;
	}
}

//===========================================================================//

void RemoteMLP::feedbackward(const float* target, float learning)
{
	// Propaga os erros na camada de saída
	outLayer->feedbackward(target, learning);

	cout << hid << ": FB Layer " << layers.size() << endl;

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
	{
		layers[i]->feedbackward(layers[i + 1]->getErrorSignal(), learning);
		cout << hid << ": FB Layer " << (i+1) << endl;
	}
}

//===========================================================================//

}
