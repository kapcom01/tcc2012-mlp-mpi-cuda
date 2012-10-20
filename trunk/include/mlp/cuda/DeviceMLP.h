#ifndef DEVICEMLP_H_
#define DEVICEMLP_H_

#include "mlp/common/MLP.h"
#include "mlp/cuda/DeviceOutLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron na GPU
 */
class DeviceMLP : public MLP
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

};

}

#endif
