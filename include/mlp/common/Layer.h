#ifndef LAYER_H_
#define LAYER_H_

#include "mlp/Types.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP
 */
class Layer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	Layer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~Layer();

	/**
	 * Randomiza os pesos de todas as conexões com a camada anterior
	 */
	virtual void randomize() = 0;

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const float* input) = 0;

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* signal, float learning) = 0;

	/**
	 * Retorna a quantidade de unidades de entrada
	 * @return Quantidade de unidades de entrada
	 */
	uint getInUnits();

	/**
	 * Retorna a quantidade de unidades de saída
	 * @return Quantidade de unidades de saída
	 */
	uint getOutUnits();

	/**
	 * Retorna o sinal funcional
	 * @return Sinal funcional
	 */
	float* getFuncSignal();

	/**
	 * Retorna o sinal de erro
	 * @return Sinal de erro
	 */
	float* getErrorSignal();

protected:

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
	 * Vetor puro de pesos e seu tamanho
	 */
	float* weights;

	/**
	 * Vetor de bias
	 */
	float* bias;

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

};

}

#endif
