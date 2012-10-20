#ifndef HOSTMLP_H_
#define HOSTMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/serial/HostOutLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron na CPU
 */
class HostMLP : public MLP
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

};

}

#endif
