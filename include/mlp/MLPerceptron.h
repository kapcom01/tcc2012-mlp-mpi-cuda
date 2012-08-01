#ifndef MULTILAYERPERCEPTRON_H_
#define MULTILAYERPERCEPTRON_H_

#include "mlp/activation/HyperbolicFunction.h"
#include "mlp/activation/LogisticFunction.h"
#include "mlp/layer/HiddenLayer.h"
#include "mlp/layer/OutputLayer.h"
#include "mlp/InputSet.h"
#include "mlp/Settings.h"

namespace MLP
{

/**
 * Classe que representa um Multi-Layer Perceptron
 */
class MLPerceptron
{

public:

	/**
	 * Constrói um MLP a partir de um conjunto de entrada
	 * @param inputSet Configurações da rede
	 */
	MLPerceptron(Settings* settings);

	/**
	 * Destrói o MLP
	 */
	~MLPerceptron();

	/**
	 * Embaralha os índices utilizando o algoritmo de Fisher-Yates
	 * @param index Vetor contendo os índices
	 * @param size Tamanho do vetor
	 */
	void shuffleIndexes(uint* indexes, uint size);

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 * @return Saída da rede
	 */
	double* feedforward(const double* input);

	/**
	 * Realiza o feedback
	 * @param expectedOutput Saída esperada
	 */
	void feedback(const double* expectedOutput);

	/**
	 * Compara a saída gerada pela rede com a saída esperada
	 * @param output Saída gerada
	 * @param expectedOutput Saída esperada
	 * @param size Tamanho das saídas
	 */
	bool compareOutput(const double* output, const double* expectedOutput,
			uint size);

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

private:

	/**
	 * Configurações da rede
	 */
	Settings* settings;

	/**
	 * Função de ativação
	 */
	ActivationFunction* activation;

	/**
	 * Taxa de aprendizado
	 */
	LearningRate* learningRate;

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
