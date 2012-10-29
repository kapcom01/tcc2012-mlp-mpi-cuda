#ifndef MLP_H_
#define MLP_H_

#include "mlp/common/OutLayer.h"
#include "mlp/common/ExampleSet.h"
#include "mlp/common/Chronometer.h"
#include "mlp/common/Indexes.h"
#include "exception/ParallelMLPException.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron
 */
class MLP
{

public:

	/**
	 * Constrói um MLP não treinado
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	MLP(v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~MLP();

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(ExampleSet &training);

protected:

	/**
	 * Linka as camadas de entrada e de saída
	 */
	void linkLayers();

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Inicializa uma operação
	 * @param set Conjunto de dados
	 */
	void initOperation(ExampleSet &set);

	/**
	 * Finaliza uma operação
	 * @param set Conjunto de dados
	 */
	void endOperation(ExampleSet &set);

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 */
	void feedforward(const float* input);

	/**
	 * Realiza o feedback
	 * @param learning Taxa de aprendizado
	 */
	void feedbackward(const float* target, float learning);

	/**
	 * Cronômetro
	 */
	Chronometer chrono;

	/**
	 * Época atual
	 */
	uint epoch;

	/**
	 * Vetor de índices
	 */
	Indexes indexes;

	/**
	 * Camadas do MLP
	 */
	vector<Layer*> layers;

	/**
	 * Primeira camada escondida
	 */
	Layer* inLayer;

	/**
	 * Camada de saída
	 */
	OutLayer* outLayer;

};

}

#endif
