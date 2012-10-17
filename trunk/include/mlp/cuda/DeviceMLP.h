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
class DeviceMLP : public MLP
{

public:

	/**
	 * Constrói um MLP que será recuperado
	 * @param mlpID ID da rede
	 */
	DeviceMLP(int mlpID);

	/**
	 * Constrói um MLP não treinado
	 * @param name Nome da rede
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	DeviceMLP(string name, v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~DeviceMLP();

	/**
	 * Adiciona uma nova camada
	 * @param inUnits Unidades de entrada
	 * @param outUnits Unidades de saída
	 * @param isOutput Indica se é uma camada de saída
	 */
	virtual void addLayer(uint inUnits, uint outUnits, bool isOutput);

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	virtual void train(DeviceExampleSet* training);

	/**
	 * Valida a rede neural
	 * @param validation Conjunto de validação
	 */
	virtual void validate(DeviceExampleSet* validation);

	/**
	 * Testa a rede neural
	 * @param test Conjunto de testes
	 */
	virtual void test(DeviceExampleSet* test);

};

}

#endif
