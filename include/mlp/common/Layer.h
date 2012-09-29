#ifndef LAYER_H_
#define LAYER_H_

#include "mlp/Vector.h"

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
	virtual void feedforward(const vec_float input) = 0;

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedback(const vec_float signal, float learning) = 0;

	/**
	 * Retorna o sinal funcional
	 * @return Sinal funcional
	 */
	vec_float getFuncSignal();

	/**
	 * Retorna o sinal de erro
	 * @return Sinal de erro
	 */
	vec_float getErrorSignal();

	/**
	 * Retorna a quantidade de entradas
	 * @param Quantidade de entradas
	 */
	uint getInUnits() const;

	/**
	 * Retorna a quantidade de neurônios
	 * @param Quantidade de neurônios
	 */
	uint getOutUnits() const;

	/**
	 * Retorna o peso do n-ésimo neurônio com a i-ésima entrada
	 * @param n Índice do neurônio
	 * @param i Índice da entrada
	 * @return Peso do n-ésimo neurônio com a i-ésima entrada
	 */
	float getWeight(uint n, uint i) const;

	/**
	 * Seta o peso do n-ésimo neurônio com a i-ésima entrada
	 * @param n Índice do neurônio
	 * @param i Índice da entrada
	 * @param weight Peso do n-ésimo neurônio com a i-ésima entrada
	 */
	void setWeight(uint n, uint i, float weight);

protected:

	/**
	 * Número de neurônios na camada anterior
	 */
	uint inUnits;

	/**
	 * Número de neurônios na camada atual
	 */
	uint outUnits;

	/**
	 * Entrada vinda da camada anterior
	 */
	vec_float input;

	/**
	 * Pesos de conexão entre os neurônios e as entradas
	 */
	hv_float weights;

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
