#include "mlp/BackpropMLP.h"

namespace MLP
{

//===========================================================================//

BackpropMLP::BackpropMLP(uint nLayers, uint* units,
		ActivationType activationType, ProblemType problemType)
{
	this->nLayers = nLayers;
	this->activationType = activationType;
	this->problemType = problemType;

	// Cria a função de ativação
	if (activationType == HYPERBOLIC)
		activation = new HyperbolicFunc();
	else
		activation = new LogisticFunc();

	// Cria a função de ativação para a camada de saída
	if (problemType == CLASSIFICATION)
		outputActivation = new StepFunc();
	else
		outputActivation = new LinearFunc();

	// Cria as camadas
	layers = new Layer*[nLayers];

	// Adiciona a primeira camada escondida
	layers[0] = new HiddenLayer(units[0], units[1], activation);

	// Adiciona as demais camadas escondidas
	uint i;
	for (i = 1; i < nLayers - 1; i++)
		layers[i] = new HiddenLayer(units[i], units[i + 1], activation);

	// Adiciona a camada de saída
	outputLayer = new OutputLayer(units[i], units[i + 1], outputActivation);
	layers[i] = outputLayer;

	// Randomiza os pesos
	randomizeWeights();
}

//===========================================================================//

BackpropMLP::~BackpropMLP()
{
	delete activation;
	delete outputActivation;
	delete[] layers;
}

//===========================================================================//

void BackpropMLP::randomizeWeights()
{
	for (uint i = 0; i < nLayers; i++)
		layers[i]->randomizeWeights();
}

//===========================================================================//

void BackpropMLP::train(InputSet* inputSet)
{
	// Cria a taxa de aprendizado
	LearningRate* learningRate = new LearningRate(inputSet->learningRate,
			inputSet->searchTime);

	// Inicializa os índices
	uint* indexes = new uint[inputSet->size];
	for (uint i = 0; i < inputSet->size; i++)
		indexes[i] = i;

	// Ciclos de treinamento
	for (uint k = 0; k < inputSet->maxIterations; k++)
	{
		cout << "Cycle: " << (k+1) << endl;

		uint hits = 0;

		// Embaralha os índices
		shuffleIndexes(indexes, inputSet->size);

		// Para cada instância
		for (uint i = 0; i < inputSet->size; i++)
		{
			uint r = indexes[i];

			cout << " |-> Instance: " << (r+1) << " ";

			// Realiza o feedforward
			const double* output = feedforward(inputSet->input[r]);

			cout << "Output: " << output[0] << " ";
			cout << "Exp output: " << inputSet->expectedOutput[r][0] << " ";

			// Verifica se acertou
			bool hit = compareOutput(output, inputSet, r);

			// Se não acertou, realiza o feedback
			if (!hit)
				feedback(inputSet->expectedOutput[r], learningRate->get());
			// Se acertou, incrementa os acertos
			else
				hits++;

			cout << endl;
		}

		// Calcula a taxa de sucesso
		inputSet->successRate = hits / (double) inputSet->size;

		cout << " |-> Sucess rate: " << inputSet->successRate << endl;

		// Se for atingido a taxa de sucesso mínima, para
		if(inputSet->successRate >= inputSet->minSuccessRate)
			break;

		// Atualiza a taxa de aprendizado
		learningRate->adjust(k);
		cout << " |-> Learning rate: " << learningRate->get() << endl << endl;
	}

	delete learningRate;
	delete[] indexes;
}

//===========================================================================//

void BackpropMLP::test(InputSet* inputSet)
{
	uint hits = 0;

	// Para cada entrada
	for (uint i = 0; i < inputSet->size; i++)
	{
		// Realiza o feedforward
		const double* output = feedforward(inputSet->input[i]);

		// Salva os valores da saída da rede neural
		for (uint j = 0; j < inputSet->outVars; j++)
			inputSet->output[i][j] = output[j];

		// Incrementa a quantidade de acertos se acertou
		hits += compareOutput(output, inputSet, i);
	}

	// Calcula a taxa de sucesso
	inputSet->successRate = inputSet->size / (double) hits;
}

//===========================================================================//

const double* BackpropMLP::feedforward(const double* input)
{
	// Propaga a entrada para a primeira camada escondida
	layers[0]->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < nLayers; i++)
		layers[i]->feedforward(layers[i - 1]->output);

	return outputLayer->output;
}

//===========================================================================//

void BackpropMLP::feedback(const double* expectedOutput, double learningRate)
{
	// Propaga a saída esperada na camada de saída
	outputLayer->feedback(expectedOutput, learningRate);

	// Propaga o sinal de feedback para o restante das camadas
	for (int i = nLayers - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->feedbackSignal, learningRate);
}

//===========================================================================//

bool BackpropMLP::compareOutput(const double* output,
		const InputSet* inputSet, uint index) const
{
	const double* expectedOutput = inputSet->expectedOutput[index];

	// Para cada saída
	for (uint i = 0; i < inputSet->outVars; i++)
		// Verifica se a diferença é maior que a tolerância máxima
		if (fabs(output[i] - expectedOutput[i]) > inputSet->maxTolerance)
			return false;

	return true;
}

//===========================================================================//

void BackpropMLP::shuffleIndexes(uint* indexes, uint size) const
{
	for (uint i = size - 1; i > 0; i--)
	{
		// Troca o valor da posição i com o de uma posição aleatória
		uint j = rand() % (i + 1);
		swap(indexes[i], indexes[j]);
	}
}

//===========================================================================//

}
