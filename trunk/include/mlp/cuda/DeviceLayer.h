#ifndef DEVICELAYER_H_
#define DEVICELAYER_H_

#include "mlp/common/Layer.h"

#define TPB 1024

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP na GPU
 */
class DeviceLayer
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
	void randomize();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	void feedforward(const float* input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedback(const float* signal, float learning);

	uint getInUnits();

	uint getOutUnits();

	float* getFuncSignal();

	float* getErrorSignal();

protected:

	/**
	 * Inicializa uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	void init(uint inUnits, uint outUnits);

	/**
	 * Quantidade de neurônios
	 */
	uint outUnits;

	/**
	 * Quantidade de entradas
	 */
	uint inUnits;

	/**
	 * Quantidade de conexões
	 */
	uint connUnits;

	/**
	 * Quantidade de blocos para execução de um kernel em função das conexões
	 */
	uint connBlocks;

	/**
	 * Quantidade de blocos para execução de um kernel em função dos neurônios
	 */
	uint outBlocks;

	/**
	 * Vetor puro de pesos e seu tamanho
	 */
	float* weights;

	/**
	 * Vetor puro do gradiente e seu tamanho
	 */
	float* gradient;

	/**
	 * Vetor puro do sinal funcional
	 */
	float* funcSignal;

	/**
	 * Vetor puro do sinal de erro
	 */
	float* errorSignal;

	/**
	 * Entrada vinda da camada anterior
	 */
	const float* input;

	/**
	 * Vetor puro de estados
	 */
	curandState* state;

};

}

#endif
