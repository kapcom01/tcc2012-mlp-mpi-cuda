#include "mlp/serial/HostMLP.h"

namespace ParallelMLP
{

//===========================================================================//

HostMLP::HostMLP(v_uint &units)
{
	// Semeia a aleatoriedade
	srand(time(NULL));

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i + 1 == units.size() - 1)
			layers.push_back(new HostOutLayer(units[i], units[i + 1]));
		else
			layers.push_back(new HostLayer(units[i], units[i + 1]));
	}

	// Seta os ponteiros para a primeira e última camada
	inLayer = layers.front();
	outLayer = (HostOutLayer*) layers.back();
	epoch = 0;

	// Randomiza os pesos
	randomize();
}

//===========================================================================//

HostMLP::~HostMLP()
{
	for (HostLayer* layer : layers)
		delete layer;
}

//===========================================================================//

void HostMLP::randomize()
{
	for (HostLayer* layer : layers)
		layer->randomize();
}

//===========================================================================//

void HostMLP::initOperation(HostExampleSet &set)
{
	// Verifica a quantidade de entradas e saídas
	if (set.getInVars() != inLayer->getInUnits())
		throw ParallelMLPException(INVALID_INPUT_VARS);
	else if (set.getOutVars() != outLayer->getOutUnits())
		throw ParallelMLPException(INVALID_OUTPUT_VARS);

	chrono.reset();
	indexes.resize(set.getSize());
	set.normalize();
	randomize();
}

//===========================================================================//

void HostMLP::endOperation(HostExampleSet &set)
{
	// Desnormaliza o conjunto de dados
	set.unnormalize();

	// Seta o erro e o tempo de execução da operação
	set.setError(outLayer->getError());
	set.setTime(chrono.getMiliseconds());
	set.setEpochs(epoch);

	cout << "totalError: " << set.getError() << endl;
	cout << "time: " << set.getTime() << endl;
}

//===========================================================================//

void HostMLP::train(HostExampleSet &training)
{
	// Inicializa a operação
	initOperation(training);

	// Épocas
	for (epoch = 0; epoch < training.getMaxEpochs(); epoch++)
	{
		cout << "Epoch " << epoch << endl;

		indexes.randomize();
		outLayer->clearError();

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes.get(i);

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

void HostMLP::feedforward(const float* input)
{
	// Propaga a entrada para a primeira camada escondida
	inLayer->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->getFuncSignal());
}

//===========================================================================//

void HostMLP::feedbackward(const float* target, float learning)
{
	// Propaga os erros na camada de saída
	outLayer->feedbackward(target, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedbackward(layers[i + 1]->getErrorSignal(), learning);
}

//===========================================================================//

}
