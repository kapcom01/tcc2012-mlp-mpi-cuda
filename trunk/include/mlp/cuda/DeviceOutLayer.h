#ifndef DEVICEOUTLAYER_H_
#define DEVICEOUTLAYER_H_

#include "mlp/common/OutLayer.h"
#include "mlp/cuda/DeviceLayer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP na GPU
 */
class DeviceOutLayer : public DeviceLayer
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
	 * @param target Saída esperada da rede neural
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedback(const float* target, float learning);

	/**
	 * Limpa o erro quadrático médio
	 */
	void clearTotalError();

	/**
	 * Retorna o erro quadrático médio
	 * @return Erro quadrático médio
	 */
	float getTotalError();

protected:

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

	float* rerror;

	/**
	 * Vetor de erros ao quadrado
	 */
	dv_float totalError;

	float* rtotalError;

	uint samples;

};

}

#endif
