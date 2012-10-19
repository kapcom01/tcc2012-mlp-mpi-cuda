#ifndef DEVICEOUTLAYER_H_
#define DEVICEOUTLAYER_H_

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
	void calculateError(const float* target);

	/**
	 * Realiza a operação de feedforward
	 * @param target Saída esperada para a rede neural
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* target, float learning);

	/**
	 * Limpa o erro quadrático médio
	 */
	void clearError();

	/**
	 * Retorna o erro quadrático médio
	 * @return Erro quadrático médio
	 */
	float getError();

protected:

	/**
	 * Vetor de erros
	 */
	float* error;

	/**
	 * Soma dos erros
	 */
	float* sum;

	/**
	 * Erro quadrático médio
	 */
	float totalError;

	/**
	 * Quantidade de amostras para o erro
	 */
	uint samples;

};

}

#endif
