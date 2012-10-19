#ifndef HOSTMLP_H_
#define HOSTMLP_H_

#include "exception/ParallelMLPException.h"
#include "mlp/common/Indexes.h"
#include "mlp/common/Chronometer.h"
#include "mlp/serial/HostOutLayer.h"
#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron na CPU
 */
class HostMLP
{

public:

	/**
	 * Constrói um MLP não treinado
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	HostMLP(v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~HostMLP();

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(HostExampleSet &training);

protected:

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Inicializa uma operação
	 * @param set Conjunto de dados
	 */
	void initOperation(HostExampleSet &set);

	/**
	 * Finaliza uma operação
	 * @param set Conjunto de dados
	 */
	void endOperation(HostExampleSet &set);

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
	vector<HostLayer*> layers;

	/**
	 * Primeira camada escondida
	 */
	HostLayer* inLayer;

	/**
	 * Camada de saída
	 */
	HostOutLayer* outLayer;

};

}

#endif
