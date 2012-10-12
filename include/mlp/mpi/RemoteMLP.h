#ifndef REMOTEMLP_H_
#define REMOTEMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/mpi/RemoteLayer.h"
#include "mlp/mpi/RemoteOutLayer.h"
#include "mlp/mpi/RemoteExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron na CPU
 */
class RemoteMLP : public MLP
{

public:

	/**
	 * Constrói um MLP que será recuperado
	 * @param mlpID ID da rede
	 */
	RemoteMLP(int mlpID);

	/**
	 * Constrói um MLP não treinado
	 * @param name Nome da rede
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	RemoteMLP(string name, v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~RemoteMLP();

	/**
	 * Adiciona uma nova camada
	 * @param inUnits Unidades de entrada
	 * @param outUnits Unidades de saída
	 * @param isOutput Indica se é camada de saída
	 */
	void addLayer(uint inUnits, uint outUnits, bool isOutput);

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(RemoteExampleSet &training);

	/**
	 * Valida a rede neural
	 * @param validation Conjunto de validação
	 */
	void validate(RemoteExampleSet &validation);

	/**
	 * Testa a rede neural
	 * @param test Conjunto de testes
	 */
	void test(RemoteExampleSet &test);

protected:

	/**
	 * ID do host atual
	 */
	uint hid;

	/**
	 * Quantidade de hosts
	 */
	uint hosts;

};

}

#endif
