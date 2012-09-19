#include "mlp/serial/BackpropMLP.h"

namespace ParallelMLP
{

//===========================================================================//

BackpropMLP::BackpropMLP(int mlpID)
{
	this->mlpID = mlpID;

	totalError = 0;
	output = NULL;
}

//===========================================================================//

BackpropMLP::BackpropMLP(string name, vector<uint> &units)
{
	this->name = name;

	// Intervalo da função de ativação hiperbólica
	range = {-1, 1};

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		LayerPtr layer(new Layer(units[i], units[i + 1]));
		layers.push_back(layer);
	}
	output = &(layers.back()->funcSignal);

	// Aloca espaço para o erro
	error.resize(units.back());

	// Randomiza os pesos
	randomize();
}

//===========================================================================//

BackpropMLP::~BackpropMLP()
{

}

//===========================================================================//

void BackpropMLP::randomize()
{
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->randomize();
}

//===========================================================================//

Range BackpropMLP::getRange() const
{
	return range;
}

//===========================================================================//

void BackpropMLP::train(HostExampleSet &training)
{
	// Randomiza os pesos e normaliza o conjunto de treinamento
	randomize();
	training.normalize();

	// Inicializa os índices
	vector<uint> indexes(training.getSize());
	initIndexes(indexes);

	// Inicializa o cronômetro
	auto begin = high_resolution_clock::now();

	// Épocas
	uint k;
	for (k = 0; k < training.maxEpochs; k++)
	{
		shuffleIndexes(indexes);
		totalError = 0;

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes[i];

			// Realiza o feedforward e salva os valores no conjunto
			feedforward(training.getInput(r));
			copyOutput(training, i);

			// Calcula o erro e realiza o feedback
			calculateError(training.getTarget(r));
			feedback(training.learning);
		}

		totalError /= training.getSize() * training.getOutVars();

		// Condição de parada: erro menor do que um valor tolerado
		if (totalError < training.tolerance)
			break;
	}

	// Duração do treinamento
	auto end = high_resolution_clock::now();

	training.time = duration_cast<milliseconds>(end - begin).count();
	training.error = totalError;

	cout << training.time << endl;

	// Desnormaliza o conjunto de treinamento
	training.unnormalize();
}

//===========================================================================//

void BackpropMLP::validate(HostExampleSet &validation)
{
	// Normaliza o conjunto de testes
	validation.normalize();

	// Inicializa o cronômetro
	auto begin = high_resolution_clock::now();

	totalError = 0;

	// Para cada entrada
	for (uint i = 0; i < validation.getSize(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(validation.getInput(i));
		copyOutput(validation, i);

		// Calcula o erro
		calculateError(validation.getTarget(i));
	}

	// Duração da validação
	auto end = high_resolution_clock::now();

	validation.time = duration_cast<milliseconds>(end - begin).count();
	validation.error = totalError;

	// Desnormaliza o conjunto de testes
	validation.unnormalize();
}

//===========================================================================//

void BackpropMLP::test(HostExampleSet &test)
{
	// Normaliza o conjunto de testes
	test.normalize();

	// Inicializa o cronômetro
	auto begin = high_resolution_clock::now();

	// Para cada entrada
	for (uint i = 0; i < test.getSize(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(test.getInput(i));
		copyOutput(test, i);
	}

	// Duração do teste
	auto end = high_resolution_clock::now();

	test.time = duration_cast<milliseconds>(end - begin).count();

	// Desnormaliza o conjunto de testes
	test.unnormalize();
}

//===========================================================================//

void BackpropMLP::copyOutput(HostExampleSet &set, uint i)
{
	set.setOutput(i, *output);
}

//===========================================================================//

void BackpropMLP::calculateError(const hv_float &target)
{
	for (uint i = 0; i < error.size(); i++)
	{
		error[i] = target[i] - (*output)[i];
		totalError += error[i] * error[i];
	}
}

//===========================================================================//

void BackpropMLP::feedforward(const hv_float &input)
{
	// Propaga a entrada para a primeira camada escondida
	layers.front()->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->funcSignal);
}

//===========================================================================//

void BackpropMLP::feedback(float learning)
{
	// Propaga os erros na camada de saída
	layers.back()->feedback(error, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->errorSignal, learning);
}

//===========================================================================//

void BackpropMLP::initIndexes(vector<uint> &indexes) const
{
	for (uint i = 0; i < indexes.size(); i++)
		indexes[i] = i;
}

//===========================================================================//

void BackpropMLP::shuffleIndexes(vector<uint> &indexes) const
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
