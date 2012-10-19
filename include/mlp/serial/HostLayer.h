#ifndef HOSTLAYER_H_
#define HOSTLAYER_H_

#include "mlp/Types.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP na CPU
 */
class HostLayer
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
	virtual void feedbackward(const float* signal, float learning);

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
	 * Retorna um valor aleatório entre -1 e 1
	 * @return Valor aleatório entre -1 e 1
	 */
	float random() const;

	/**
	 * Ativa um valor através da função hiperbólica
	 * @param x Valor a ser ativado
	 * @return Valor ativado
	 */
	float activate(float x) const;

	/**
	 * Desativa um valor através da derivada da função hiperbólica
	 * @param y Valor ativado
	 * @return Valor desativado
	 */
	float derivate(float y) const;

	/**
	 * Quantidade de unidades de entrada
	 */
	uint inUnits;

	/**
	 * Quantidade de unidades de saída
	 */
	uint outUnits;

	/**
	 * Quantidade de unidades de conexões
	 */
	uint connUnits;

	/**
	 * Vetor de entrada
	 */
	const float* input;

	/**
	 * Vetor puro de pesos
	 */
	float* weights;

	/**
	 * Gradiente dos neurônios
	 */
	float* gradient;

	/**
	 * Sinal funcional dos neurônios
	 */
	float* funcSignal;

	/**
	 * Sinal de erro
	 */
	float* errorSignal;

};

}

#endif
