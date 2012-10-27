#ifndef DEVICELAYER_H_
#define DEVICELAYER_H_

#include "mlp/common/Layer.h"
#include "mlp/cuda/CUDATypes.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP na GPU
 */
class DeviceLayer : public virtual Layer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	DeviceLayer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~DeviceLayer();

	/**
	 * Randomiza os pesos de todas as conexões com a camada anterior
	 */
	virtual void randomize();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const float* input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* signal, float learning);

protected:

	/**
	 * Quantidade de blocos para execução de um kernel em função das conexões
	 */
	uint connBlocks;

	/**
	 * Quantidade de blocos para execução de um kernel em função dos neurônios
	 */
	uint outBlocks;

	/**
	 * Vetor puro de estados
	 */
	curandState* state;

};

}

#endif
