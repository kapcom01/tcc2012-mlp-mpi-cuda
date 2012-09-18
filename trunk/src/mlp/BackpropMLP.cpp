#include "mlp/BackpropMLP.h"

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

BackpropMLP::BackpropMLP(string name, vuint &units)
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

void BackpropMLP::train(ExampleSet &training)
{
	// Randomiza os pesos e normaliza o conjunto de treinamento
	randomize();
	training.normalize();

	// Inicializa os índices
	vuint indexes(training.size());
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
		for (uint i = 0; i < training.size(); i++)
		{
			uint r = indexes[i];

			// Realiza o feedforward e salva os valores no conjunto
			feedforward(training.input[r]);
			copyOutput(training, i);

			// Calcula o erro e realiza o feedback
			calculateError(training.target[r]);
			feedback(training.learning);
		}

		totalError /= training.size() * training.outVars();

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

void BackpropMLP::validate(ExampleSet &validation)
{
	// Normaliza o conjunto de testes
	validation.normalize();

	// Inicializa o cronômetro
	auto begin = high_resolution_clock::now();

	totalError = 0;

	// Para cada entrada
	for (uint i = 0; i < validation.size(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(validation.input[i]);
		copyOutput(validation, i);

		// Calcula o erro
		calculateError(validation.target[i]);
	}

	// Duração da validação
	auto end = high_resolution_clock::now();

	validation.time = duration_cast<milliseconds>(end - begin).count();
	validation.error = totalError;

	// Desnormaliza o conjunto de testes
	validation.unnormalize();
}

//===========================================================================//

void BackpropMLP::test(ExampleSet &test)
{
	// Normaliza o conjunto de testes
	test.normalize();

	// Inicializa o cronômetro
	auto begin = high_resolution_clock::now();

	// Para cada entrada
	for (uint i = 0; i < test.size(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(test.input[i]);
		copyOutput(test, i);
	}

	// Duração do teste
	auto end = high_resolution_clock::now();

	test.time = duration_cast<milliseconds>(end - begin).count();

	// Desnormaliza o conjunto de testes
	test.unnormalize();
}

//===========================================================================//

void BackpropMLP::copyOutput(ExampleSet &set, uint i)
{
	set.output[i] = (*output);
//	copy(output->begin(), output->end(), set.output[i].begin());
}

//===========================================================================//

void BackpropMLP::calculateError(const vdouble &target)
{
	for (uint i = 0; i < error.size(); i++)
	{
		error[i] = target[i] - (*output)[i];
		totalError += error[i] * error[i];
	}
}

//===========================================================================//

void BackpropMLP::feedforward(const vdouble &input)
{
	// Propaga a entrada para a primeira camada escondida
	layers.front()->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->funcSignal);
}

//===========================================================================//

void BackpropMLP::feedback(double learning)
{
	// Propaga os erros na camada de saída
	layers.back()->feedback(error, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->errorSignal, learning);
}

//===========================================================================//

void BackpropMLP::initIndexes(vuint &indexes) const
{
	for (uint i = 0; i < indexes.size(); i++)
		indexes[i] = i;
}

//===========================================================================//

void BackpropMLP::shuffleIndexes(vuint &indexes) const
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
