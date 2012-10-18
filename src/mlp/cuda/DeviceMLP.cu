#include "mlp/cuda/DeviceMLP.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceMLP::DeviceMLP(v_uint &units)
{
	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i == units.size() - 2)
			layers.push_back(new DeviceOutLayer(units[i], units[i + 1]));
		else
			layers.push_back(new DeviceLayer(units[i], units[i + 1]));
	}

	inLayer = layers.front();
	outLayer = (DeviceOutLayer*) layers.back();

	randomize();
}

//===========================================================================//

DeviceMLP::~DeviceMLP()
{

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
		throw runtime_error("invalid input vars");
	else if (set.getOutVars() != outLayer->getOutUnits())
		throw runtime_error("invalid output vars");

	// Reseta o cronômetro, o erro total e a época
	chrono.reset();
	epoch = 0;

	// Normaliza o conjunto de dados
	set.normalize();

	// Se for treinamento, randomiza os pesos e inicializa os índices
	if (set.isTraining())
	{
		randomize();
		initIndexes(set.getSize());
	}
}

//===========================================================================//

void DeviceMLP::endOperation(DeviceExampleSet &set)
{
	// Desnormaliza o conjunto de dados
	set.unnormalize();

	// Seta o erro e o tempo de execução da operação
	set.setError(outLayer->getTotalError());
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
	for (; epoch < training.getMaxEpochs(); epoch++)
	{
		cout << "Epoch " << epoch << endl;

		shuffleIndexes();
		outLayer->clearTotalError();

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes[i];

			// Realiza o feedforward e salva os valores no conjunto
			feedforward(training.getInput(r).data());
			//training.setOutput(r, outLayer->getFuncSignal());

			// Realiza o feedback
			feedback(training.getTarget(r).data(), training.getLearning());
		}

		// Condição de parada: erro menor do que um valor tolerado
		if (outLayer->getTotalError() < training.getTolerance())
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

void DeviceMLP::feedback(const float* target, float learning)
{
	// Propaga os erros na camada de saída
	outLayer->feedback(target, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->getErrorSignal(), learning);
}

//===========================================================================//

void DeviceMLP::initIndexes(uint size)
{
	indexes.resize(size);
	for (uint i = 0; i < indexes.size(); i++)
		indexes[i] = i;
}

//===========================================================================//

void DeviceMLP::shuffleIndexes()
{
	for (uint i = indexes.size() - 1; i > 0; i--)
	{
		// Troca o valor da posição i com o de uma posição aleatória
		uint j = rand() % (i + 1);
		swap(indexes[i], indexes[j]);
	}
}

//===========================================================================//


}
