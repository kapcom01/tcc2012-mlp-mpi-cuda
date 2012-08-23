#include "mlp/BackpropMLP.h"

namespace MLP
{

//===========================================================================//

BackpropMLP::BackpropMLP(vector<uint> &units, ActivationType activationType)
{
	// Constrói a função de ativação
	if (activationType == LOGISTIC)
		activation = ActivationFuncPtr(new LogisticFunc);
	else
		activation = ActivationFuncPtr(new HyperbolicFunc);

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
		layers.push_back(LayerPtr(
				new Layer(units[i], units[i + 1], *activation)));

	// Aloca espaço para o erro
	error.resize(units[layers.size()]);

	// Randomiza os pesos
	randomizeWeights();

	for (uint i = 0; i < layers.size(); i++)
	{
		cout << "Layer: " << (i + 1) << endl;
		for (uint j = 0; j < layers[i]->outUnits; j++)
		{
			cout << "Neuron: " << (j + 1) << " |";
			for (uint k = 0; k <= layers[i]->inUnits; k++)
				cout << " " << layers[i]->weights[j][k];
			cout << endl;
		}
	}
}

//===========================================================================//

BackpropMLP::~BackpropMLP()
{

}

//===========================================================================//

void BackpropMLP::randomizeWeights()
{
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->randomizeWeights();
}

//===========================================================================//

void BackpropMLP::train(ExampleSet &trainingSet)
{
	// Inicializa os índices
	vector<uint> indexes;
	for (uint i = 0; i < trainingSet.size(); i++)
		indexes.push_back(i);

	// Ciclos de treinamento
	for (uint k = 0; k < trainingSet.maxEpochs; k++)
	{
		cout << "Epoch: " << (k+1) << endl;

		uint hits = 0;

		// Embaralha os índices
		shuffleIndexes(indexes);

		// Para cada instância
		for (uint i = 0; i < trainingSet.size(); i++)
		{
			uint r = indexes[i];

			cout << " |-> Instance: " << (r+1) << " ";

			// Realiza o feedforward
			const vector<double> &output = feedforward(trainingSet.input[r]);

			cout << "Input: " << trainingSet.input[r][0] << " "
					<< trainingSet.input[r][1] << " ";
			cout << "Target: " << trainingSet.target[r][0] << " ";
			cout << "Output: " << output[0] << " ";

			// Verifica se acertou
			bool hit = compareOutput(output, trainingSet.target[r],
					trainingSet.maxTolerance);

			// Se não acertou, realiza o feedback
			if (!hit)
				feedback(trainingSet.target[r], trainingSet.learningRate,
						trainingSet.momentum);
			// Se acertou, incrementa os acertos
			else
				hits++;

			cout << endl;
		}

		// Calcula a taxa de sucesso
		trainingSet.successRate = hits / (double) trainingSet.size();

		cout << " |-> Sucess rate: " << trainingSet.successRate << endl;

		// Se for atingido a taxa de sucesso mínima, para
		if(trainingSet.successRate >= trainingSet.minSuccessRate)
			break;
	}
}

//===========================================================================//

void BackpropMLP::test(ExampleSet &testSet)
{
	uint hits = 0;

	// Para cada entrada
	for (uint i = 0; i < testSet.size(); i++)
	{
		// Realiza o feedforward
		const vector<double> &output = feedforward(testSet.input[i]);

		// Salva os valores da saída da rede neural
		for (uint j = 0; j < testSet.outVars(); j++)
			testSet.output[i][j] = output[j];

		// Incrementa a quantidade de acertos se acertou
		hits += compareOutput(output, testSet.output[i], testSet.maxTolerance);
	}

	// Calcula a taxa de sucesso
	testSet.successRate = testSet.size() / (double) hits;
}

//===========================================================================//

const vector<double>& BackpropMLP::feedforward(const vector<double> &input)
{
	// Propaga a entrada para a primeira camada escondida
	layers.front()->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->funcSignal);

	return layers.back()->funcSignal;
}

//===========================================================================//

void BackpropMLP::feedback(const vector<double> &target, double learningRate,
		double momentum)
{
	const vector<double> &output = layers.back()->funcSignal;

	// Calcula o erro cometido pela rede
	for (uint i = 0; i < error.size(); i++)
		error[i] = target[i] - output[i];

	// Propaga o erro na camada de saída
	layers.back()->feedback(error, learningRate, momentum);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->errorSignal, learningRate,
				momentum);
}

//===========================================================================//

bool BackpropMLP::compareOutput(const vector<double> output,
		const vector<double> &target, double maxTolerance) const
{
	// Para cada saída
	for (uint i = 0; i < output.size(); i++)
		// Verifica se a diferença é maior que a tolerância máxima
		if (fabs(output[i] - target[i]) > maxTolerance)
			return false;

	return true;
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
