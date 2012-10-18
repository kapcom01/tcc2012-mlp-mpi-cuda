#ifndef DEVICEMLP_H_
#define DEVICEMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/cuda/DeviceOutLayer.h"
#include "mlp/cuda/DeviceExampleSet.h"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

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

	void randomize();

	void initOperation(DeviceExampleSet &training);

	void endOperation(DeviceExampleSet &training);

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(DeviceExampleSet &training);

private:

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 */
	void feedforward(const float* input);

	/**
	 * Realiza o feedback
	 * @param learning Taxa de aprendizado
	 */
	void feedback(const float* target, float learning);

	/**
	 * Inicializa os índices
	 * @param size Tamanho do vetor
	 */
	void initIndexes(uint size);

	/**
	 * Embaralha os índices utilizando o algoritmo de Fisher-Yates
	 */
	void shuffleIndexes();

	/**
	 * Cronômetro
	 */
	Chronometer chrono;

	uint epoch;

	/**
	 * Vetor de índices para o treinamento
	 */
	v_uint indexes;

	vector<DeviceLayer*> layers;

	DeviceLayer* inLayer;

	DeviceOutLayer* outLayer;

};

}

#endif
