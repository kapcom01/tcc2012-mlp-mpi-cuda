#ifndef HOSTMLP_H_
#define HOSTMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/serial/HostLayer.h"
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
	HostMLP(string name, vector<uint> &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~HostMLP();

	/**
	 * Adiciona uma nova camada
	 * @param inUnits Unidades de entrada
	 * @param outUnits Unidades de saída
	 */
	void addLayer(uint inUnits, uint outUnits);

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(HostExampleSet &training);

	/**
	 * Valida a rede neural
	 * @param validation Conjunto de validação
	 */
	void validate(HostExampleSet &validation);

	/**
	 * Testa a rede neural
	 * @param test Conjunto de testes
	 */
	void test(HostExampleSet &test);

};

}

#endif
