#include "mlp/common/MLP.h"

namespace ParallelMLP
{

//===========================================================================//

MLP::MLP(int mlpID)
{
	this->mlpID = mlpID;
	this->totalError = 0;
	this->epoch = 0;
}

//===========================================================================//

MLP::MLP(string name, vector<uint> &units)
{
	this->name = name;
	this->mlpID = -1;
	this->totalError = 0;
	this->epoch = 0;

	// Intervalo da função de ativação hiperbólica
	range.lower = -1, range.upper = 1;
}

//===========================================================================//

MLP::~MLP()
{
	for (uint i = 0; i < layers.size(); i++)
		delete layers[i];
}

//===========================================================================//

void MLP::randomize()
{
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->randomize();
}

//===========================================================================//

void MLP::initOperation(ExampleSet &set)
{
	// Reseta o cronômetro, o erro total e a época
	chrono.reset();
	totalError = 0;
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

void MLP::endOperation(ExampleSet &set)
{
	// Desnormaliza o conjunto de dados
	set.unnormalize();

	// Seta o erro e o tempo de execução da operação
	set.setError(totalError);
	set.setTime(chrono.getMiliseconds());
	set.setEpochs(epoch);
}

//===========================================================================//

void MLP::calculateError(const vec_float target)
{
	for (uint i = 0; i < error.size(); i++)
	{
		error[i] = target[i] - output[i];
		totalError += error[i] * error[i];
	}
}

//===========================================================================//

void MLP::feedforward(const vec_float input)
{
	// Propaga a entrada para a primeira camada escondida
	layers.front()->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->getFuncSignal());
}

//===========================================================================//

void MLP::feedback(float learning)
{
	// Propaga os erros na camada de saída
	layers.back()->feedback(error, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->getErrorSignal(), learning);
}

//===========================================================================//

void MLP::initIndexes(uint size)
{
	indexes.resize(size);
	for (uint i = 0; i < indexes.size(); i++)
		indexes[i] = i;
}

//===========================================================================//

void MLP::shuffleIndexes()
{
	for (uint i = indexes.size() - 1; i > 0; i--)
	{
		// Troca o valor da posição i com o de uma posição aleatória
		uint j = rand() % (i + 1);
		swap(indexes[i], indexes[j]);
	}
}

//===========================================================================//

int MLP::getID() const
{
	return mlpID;
}

//===========================================================================//

void MLP::setID(int mlpID)
{
	this->mlpID = mlpID;
}

//===========================================================================//

string MLP::getName() const
{
	return name;
}

//===========================================================================//

void MLP::setName(string name)
{
	this->name = name;
}

//===========================================================================//

Range MLP::getRange() const
{
	return range;
}

//===========================================================================//

void MLP::setRange(Range range)
{
	this->range = range;
}

//===========================================================================//

uint MLP::getNLayers() const
{
	return layers.size();
}

//===========================================================================//

Layer& MLP::getLayer(uint i)
{
	return *(layers[i]);
}

//===========================================================================//

const Layer& MLP::getLayer(uint i) const
{
	return *(layers[i]);
}

//===========================================================================//

void MLP::setOutput()
{
	Layer *last = layers.back();
	output = last->getFuncSignal();
	error.resize(last->getOutUnits());
}

//===========================================================================//

}
