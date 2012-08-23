#ifndef BACKPROPMLP_H_
#define BACKPROPMLP_H_

#include "mlp/activation/HyperbolicFunc.h"
#include "mlp/activation/LogisticFunc.h"
#include "mlp/Layer.h"
#include "mlp/ExampleSet.h"

namespace Database { class MLPHelper; }

namespace MLP
{

/**
 * Classe que representa um Multi-Layer Perceptron
 */
class BackpropMLP
{

public:

	/**
	 * Constrói um MLP não treinado
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 * @param activationType Tipo da função de ativação
	 */
	BackpropMLP(vector<uint> &units, ActivationType activationType);

	/**
	 * Destrói o MLP
	 */
	~BackpropMLP();

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomizeWeights();

	/**
	 * Treina a rede neural
	 * @param trainingSet Conjunto de treinamento
	 */
	void train(ExampleSet &trainingSet);

	/**
	 * Testa a rede neural
	 * @param testSet Conjunto de testes
	 */
	void test(ExampleSet &testSet);

	friend class Database::MLPHelper;

private:

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 * @return Saída da rede
	 */
	const vector<double>& feedforward(const vector<double> &input);

	/**
	 * Realiza o feedback
	 * @param target Saída alvo esperada
	 * @param learningRate Taxa de aprendizado
	 * @param momentum Momento
	 */
	void feedback(const vector<double> &target, double learningRate,
			double momentum);

	/**
	 * Compara a saída gerada pela rede com a saída esperada
	 * @param output Saída gerada
	 * @param target Saída alvo esperada
	 */
	bool compareOutput(const vector<double> output,
			const vector<double> &target, double maxTolerance) const;

	/**
	 * Embaralha os índices utilizando o algoritmo de Fisher-Yates
	 * @param index Vetor contendo os índices
	 */
	void shuffleIndexes(vector<uint> &indexes) const;

	/**
	 * ID da rede
	 */
	int mlpID;

	/**
	 * Função de ativação
	 */
	ActivationFuncPtr activation;

	/**
	 * Camadas
	 */
	vector<LayerPtr> layers;

	/**
	 * Erro cometido pela rede
	 */
	vector<double> error;

};

}

#endif
