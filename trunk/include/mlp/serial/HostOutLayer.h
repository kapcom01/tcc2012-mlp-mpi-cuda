#ifndef HOSTOUTLAYER_H_
#define HOSTOUTLAYER_H_

#include "mlp/common/OutLayer.h"
#include "mlp/serial/HostLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP na CPU
 */
class HostOutLayer : public OutLayer, public HostLayer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	HostOutLayer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~HostOutLayer();

	/**
	 * Calcula o erro da rede
	 * @param target Saída esperada para a rede neural
	 */
	virtual void calculateError(const float* target);

	/**
	 * Realiza a operação de feedforward
	 * @param target Saída esperada para a rede neural
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* target, float learning);

};

}

#endif
