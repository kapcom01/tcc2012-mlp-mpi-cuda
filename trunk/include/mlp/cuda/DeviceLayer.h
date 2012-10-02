#ifndef DEVICELAYER_H_
#define DEVICELAYER_H_

#include "mlp/common/Layer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP na GPU
 */
class DeviceLayer : public Layer
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
	 * Inicia uma operação
	 */
	void initOperation();

	/**
	 * Finaliza uma operação
	 */
	void endOperation();

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

protected:

	/**
	 * Copia os dados da memória da CPU para a memória da GPU
	 */
	void copyToDevice();

	/**
	 * Copia os dados da memória da GPU para a memória da CPU
	 */
	void copyToHost();

	/**
	 * Pesos de conexão entre os neurônios e as entradas
	 */
	dv_float devWeights;

	/**
	 * Vetor puro de pesos e seu tamanho
	 */
	vec_float rawWeights;

	/**
	 * Gradiente dos neurônios
	 */
	dv_float devGradient;

	/**
	 * Vetor puro do gradiente e seu tamanho
	 */
	vec_float rawGradient;

	/**
	 * Sinal funcional dos neurônios
	 */
	dv_float devFuncSignal;

	/**
	 * Sinal de erro
	 */
	dv_float devErrorSignal;

	/**
	 * Estados para geração de números aleatórios
	 */
	dv_rand devState;

	/**
	 * Vetor puro de estados e seu tamanho
	 */
	vec_rand rawState;

};

}

#endif
