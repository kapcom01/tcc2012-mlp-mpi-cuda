#ifndef HOSTMLP_H_
#define HOSTMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/serial/HostOutLayer.h"
#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron na CPU
 */
class HostMLP : public MLP
{

public:

	/**
	 * Constrói um MLP que será recuperado
	 * @param mlpID ID da rede
	 */
	HostMLP(int mlpID);

	/**
	 * Constrói um MLP não treinado
	 * @param name Nome da rede
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	HostMLP(string name, v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~HostMLP();

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
	virtual void train(HostExampleSet* training);

	/**
	 * Valida a rede neural
	 * @param validation Conjunto de validação
	 */
	virtual void validate(HostExampleSet* validation);

	/**
	 * Testa a rede neural
	 * @param test Conjunto de testes
	 */
	virtual void test(HostExampleSet* test);

};

}

#endif
