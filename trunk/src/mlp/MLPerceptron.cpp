#include "mlp/MLPerceptron.h"

namespace MLP
{

//===========================================================================//

MLPerceptron::MLPerceptron(Settings* settings)
{
	this->settings = settings;

	// Cria a função de ativação
	if (settings->activationType == HYPERBOLIC)
		activation = new HyperbolicFunction();
	else
		activation = new LogisticFunction();

	// Cria a taxa de aprendizado
	learningRate = new LearningRate(settings->initialLR, settings->minLR,
			settings->maxLR);

	// Cria as camadas
	layers = new Layer*[settings->nLayers];
	uint* units = settings->units;

	// Adiciona a primeira camada escondida
	layers[0] = new HiddenLayer(units[0], units[1], activation, learningRate);

	// Adiciona as demais camadas escondidas
	uint i;
	for (i = 1; i < settings->nLayers - 1; i++)
		layers[i] = new HiddenLayer(units[i], units[i + 1], activation,
				learningRate);

	// Adiciona a camada de saída
	outputLayer = new OutputLayer(units[i], units[i + 1], activation,
			learningRate);
	layers[i] = outputLayer;
}

//===========================================================================//

MLPerceptron::~MLPerceptron()
{
	delete activation;
	delete learningRate;
	delete[] layers;
}

//===========================================================================//

void MLPerceptron::shuffleIndexes(uint* indexes, uint size)
{
	for (uint i = size - 1; i > 0; i--)
	{
		// Troca o valor da posição i com o de uma posição aleatória
		uint j = rand() % (i + 1);
		swap(indexes[i], indexes[j]);
	}
}

//===========================================================================//

double* MLPerceptron::feedforward(const double* input)
{
	// Propaga a entrada para a primeira camada escondida
	layers[0]->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < settings->nLayers; i++)
		layers[i]->feedforward(layers[i - 1]->getOutput());

	return outputLayer->getOutput();
}

//===========================================================================//

void MLPerceptron::feedback(const double* expectedOutput)
{
	// Propaga a saída esperada na camada de saída
	outputLayer->feedback(expectedOutput);

	// Propaga o sinal de feedback para o restante das camadas
	for (int i = settings->nLayers - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->getFeedback());
}

//===========================================================================//

bool MLPerceptron::compareOutput(const double* output,
		const double* expectedOutput, uint size)
{
	// Para cada saída
	for (uint i = 0; i < size; i++)
		// Verifica se a diferença é maior que a tolerância máxima
		if (fabs(output[i] - expectedOutput[i]) > settings->maxTolerance)
			return false;

	return true;
}

//===========================================================================//

void MLPerceptron::train(InputSet* inputSet)
{
	// Inicializa os índices
	uint* indexes = new uint[inputSet->size];
	for (uint i = 0; i < inputSet->size; i++)
		indexes[i] = i;

	// Ciclos de treinamento
	for (uint k = 0; ; k++)
	{
		uint hits = 0;

		// Embaralha os índices
		shuffleIndexes(indexes, inputSet->size);

		// Para cada instância
		for (uint i = 0; i < inputSet->size; i++)
		{
			uint r = indexes[i];

			// Realiza o feedforward
			double* output = feedforward(inputSet->input[r]);

			// Verifica se acertou
			bool hit = compareOutput(output, inputSet->expectedOutput[r],
					inputSet->outVars);

			// Se não acertou, realiza o feedback
			if (!hit)
				feedback(inputSet->expectedOutput[r]);
			// Se acertou, incrementa os acertos
			else
				hits++;

			// Atualiza a taxa de aprendizado
			outputLayer->updateLearningRate(inputSet->expectedOutput[r]);
		}

		// Calcula a taxa de sucesso
		inputSet->successRate = hits / (double) inputSet->size;

		// Se for atingido a taxa de sucesso mínima, para
		if(inputSet->successRate >= settings->minSuccessRate)
			break;
	}

	delete[] indexes;
}

//===========================================================================//

void MLPerceptron::test(InputSet* inputSet)
{
	uint hits = 0;

	// Para cada entrada
	for (uint i = 0; i < inputSet->size; i++)
	{
		// Realiza o feedforward
		double* output = feedforward(inputSet->input[i]);

		// Salva os valores da saída da rede neural
		for (uint j = 0; j < inputSet->outVars; j++)
			inputSet->output[i][j] = output[j];

		// Incrementa a quantidade de acertos se acertou
		hits += compareOutput(output, inputSet->expectedOutput[i],
				inputSet->outVars);
	}

	// Calcula a taxa de sucesso
	inputSet->successRate = inputSet->size / (double) hits;
}

//===========================================================================//

}
