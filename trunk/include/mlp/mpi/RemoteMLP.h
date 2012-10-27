#ifndef REMOTEMLP_H_
#define REMOTEMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/mpi/RemoteLayer.h"
#include "mlp/mpi/RemoteOutLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron em várias CPUs
 */
class RemoteMLP : public MLP
{

public:

	/**
	 * Constrói um MLP não treinado
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	RemoteMLP(v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~RemoteMLP();

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	virtual void train(ExampleSet &training);

protected:

	/**
	 * ID do host
	 */
	uint hid;

	/**
	 * Quantidade de hosts
	 */
	uint hosts;

};

}

#endif
