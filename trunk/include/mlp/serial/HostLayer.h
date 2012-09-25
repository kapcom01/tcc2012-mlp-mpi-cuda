#ifndef HOSTLAYER_H_
#define HOSTLAYER_H_

#include "mlp/common/Layer.h"
#include "mlp/serial/HostNeuron.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP na CPU
 */
class HostLayer : public Layer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	HostLayer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~HostLayer();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	void feedforward(const vec_float input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	void feedback(const vec_float signal, float learning);

};

}

#endif
