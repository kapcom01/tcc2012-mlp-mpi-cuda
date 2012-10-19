#ifndef DEVICEMLP_H_
#define DEVICEMLP_H_

#include "exception/ParallelMLPException.h"
#include "mlp/common/Indexes.h"
#include "mlp/common/Chronometer.h"
#include "mlp/cuda/DeviceOutLayer.h"
#include "mlp/cuda/DeviceExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron na GPU
 */
class DeviceMLP
{

public:

	/**
	 * Constrói um MLP não treinado
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	DeviceMLP(v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~DeviceMLP();

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(DeviceExampleSet &training);

protected:

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Inicializa uma operação
	 * @param set Conjunto de dados
	 */
	void initOperation(DeviceExampleSet &set);

	/**
	 * Finaliza uma operação
	 * @param set Conjunto de dados
	 */
	void endOperation(DeviceExampleSet &set);

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
	vector<DeviceLayer*> layers;

	/**
	 * Primeira camada escondida
	 */
	DeviceLayer* inLayer;

	/**
	 * Camada de saída
	 */
	DeviceOutLayer* outLayer;

};

}

#endif
