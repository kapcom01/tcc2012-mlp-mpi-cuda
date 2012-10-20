#ifndef DEVICEOUTLAYER_H_
#define DEVICEOUTLAYER_H_

#include "mlp/common/OutLayer.h"
#include "mlp/cuda/DeviceLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP na GPU
 */
class DeviceOutLayer : public OutLayer, public DeviceLayer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	DeviceOutLayer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~DeviceOutLayer();

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

protected:

	/**
	 * Soma dos erros
	 */
	float* sum;

};

}

#endif
