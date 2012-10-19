#include "mlp/cuda/DeviceMLP.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceMLP::DeviceMLP(v_uint &units)
{
	// Semeia a aleatoriedade
	srand(time(NULL));

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i + 1 == units.size() - 1)
			layers.push_back(new DeviceOutLayer(units[i], units[i + 1]));
		else
			layers.push_back(new DeviceLayer(units[i], units[i + 1]));
	}

	// Seta os ponteiros para a primeira e última camada
	inLayer = layers.front();
	outLayer = (DeviceOutLayer*) layers.back();
	epoch = 0;

	// Randomiza os pesos
	randomize();
}

//===========================================================================//

DeviceMLP::~DeviceMLP()
{
	for (uint i = 0; i < layers.size(); i++)
		delete layers[i];
}

//===========================================================================//

void DeviceMLP::randomize()
{
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->randomize();
}

//===========================================================================//

void DeviceMLP::initOperation(DeviceExampleSet &set)
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

void DeviceMLP::endOperation(DeviceExampleSet &set)
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

void DeviceMLP::train(DeviceExampleSet &training)
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

void DeviceMLP::feedforward(const float* input)
{
	// Propaga a entrada para a primeira camada escondida
	inLayer->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->getFuncSignal());
}

//===========================================================================//

void DeviceMLP::feedbackward(const float* target, float learning)
{
	// Propaga os erros na camada de saída
	outLayer->feedbackward(target, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedbackward(layers[i + 1]->getErrorSignal(), learning);
}

//===========================================================================//

}
