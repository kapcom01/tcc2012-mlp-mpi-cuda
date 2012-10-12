#ifndef DEVICEOUTLAYER_H_
#define DEVICEOUTLAYER_H_

#include "mlp/common/OutLayer.h"
#include "mlp/cuda/DeviceLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP na GPU
 */
class DeviceOutLayer : public DeviceLayer, public OutLayer
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
	void calculateError(const vec_float &target);

	/**
	 * Realiza a operação de feedforward
	 * @param target Saída esperada da rede neural
	 * @param learning Taxa de aprendizado
	 */
	void feedback(const vec_float &target, float learning);

protected:

	/**
	 * Constrói uma camada vazia
	 */
	DeviceOutLayer();

	/**
	 * Inicializa uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	void init(uint inUnits, uint outUnits);

	/**
	 * Vetor de erros
	 */
	dv_float error;

};

}

#endif
