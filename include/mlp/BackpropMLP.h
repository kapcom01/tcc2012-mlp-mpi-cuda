#ifndef MULTILAYERPERCEPTRON_H_
#define MULTILAYERPERCEPTRON_H_

#include "mlp/activation/HyperbolicFunc.h"
#include "mlp/activation/LogisticFunc.h"
#include "mlp/layer/HiddenLayer.h"
#include "mlp/layer/OutputLayer.h"
#include "mlp/InputSet.h"

namespace Database { class MLPHelper; }

namespace MLP
{

/**
 * Tipos de função de ativação
 */
enum ActivationType
{
	HYPERBOLIC = 1, LOGISTIC = 2
};

/**
 * Tipos de problemas
 */
enum ProblemType
{
	CLASSIFICATION = 1, APROXIMATION = 2
};

/**
 * Classe que representa um Multi-Layer Perceptron
 */
class BackpropMLP
{

public:

	/**
	 * Constrói um MLP a partir de um conjunto de entrada
	 * @param nLayers Quantidade de camadas
	 * @param units Quantidade de neurônios por camada
	 * @param activationType Tipo da função de ativação
	 * @param problemType Tipo do problema
	 */
	BackpropMLP(uint nLayers, uint* units, ActivationType activationType,
			ProblemType problemType);

	/**
	 * Destrói o MLP
	 */
	~BackpropMLP();

	/**
	 * Randomiza os pesos
	 */
	void randomizeWeights();

	/**
	 * Treina a rede neural
	 * @param inputSet Conjunto de entradas
	 */
	void train(InputSet* inputSet);

	/**
	 * Testa a rede neural
	 * @param inputSet Conjunto de entradas
	 */
	void test(InputSet* inputSet);

	friend class Database::MLPHelper;

private:

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 * @return Saída da rede
	 */
	const double* feedforward(const double* input);

	/**
	 * Realiza o feedback
	 * @param expectedOutput Saída esperada
	 * @param learningRate Taxa de aprendizado
	 */
	void feedback(const double* expectedOutput, double learningRate);

	/**
	 * Compara a saída gerada pela rede com a saída esperada
	 * @param output Saída gerada
	 * @param inputSet Conjunto de entradas
	 * @param index Índice da instância
	 */
	bool compareOutput(const double* output, const InputSet* inputSet,
			uint index) const;

	/**
	 * Embaralha os índices utilizando o algoritmo de Fisher-Yates
	 * @param index Vetor contendo os índices
	 * @param size Tamanho do vetor
	 */
	void shuffleIndexes(uint* indexes, uint size) const;

	/**
	 * ID da rede
	 */
	int mlpID;

	/**
	 * Quantidade de camadas
	 */
	uint nLayers;

	/**
	 * Tipo da função de ativação
	 */
	ActivationType activationType;

	/**
	 * Tipo do problema
	 */
	ProblemType problemType;

	/**
	 * Função de ativação
	 */
	ActivationFunc* activation;

	/**
	 * Função de ativação na camada de saída
	 */
	ActivationFunc* outputActivation;

	/**
	 * Camadas
	 */
	Layer** layers;

	/**
	 * Camada de saída
	 */
	OutputLayer* outputLayer;

};

}

#endif
