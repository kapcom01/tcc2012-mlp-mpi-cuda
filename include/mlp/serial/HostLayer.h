#ifndef HOSTLAYER_H_
#define HOSTLAYER_H_

#include "mlp/common/Layer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP na CPU
 */
class HostLayer : virtual public Layer
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
	 * Randomiza os pesos de todas as conexões com a camada anterior
	 */
	virtual void randomize();

	/**
	 * Inicia uma operação
	 */
	virtual void initOperation();

	/**
	 * Finaliza uma operação
	 */
	virtual void endOperation();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const vec_float &input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedback(const vec_float &signal, float learning);

protected:

	/**
	 * Constrói uma camada vazia
	 */
	HostLayer();

	/**
	 * Inicializa uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	void init(uint inUnits, uint outUnits);

	/**
	 * Vetor puro de pesos
	 */
	vec_float rawWeights;

	/**
	 * Gradiente dos neurônios
	 */
	hv_float gradient;

	/**
	 * Sinal funcional dos neurônios
	 */
	hv_float funcSignal;

	/**
	 * Sinal de erro
	 */
	hv_float errorSignal;

};

}

#endif
