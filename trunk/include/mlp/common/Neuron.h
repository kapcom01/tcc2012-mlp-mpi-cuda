#ifndef NEURON_H_
#define NEURON_H_

#include "mlp/Vector.h"
#include <cmath>

namespace ParallelMLP
{

/**
 * Classe que representa um neurônio
 */
class Neuron
{

public:

	/**
	 * Constrói um neurônio
	 * @param inUnits Quantidade de entradas
	 */
	Neuron(uint inUnits);

	/**
	 * Destrói o neurônio
	 */
	virtual ~Neuron();

	/**
	 * Randomiza os pesos das conexões
	 */
	virtual void randomize() = 0;

	/**
	 * Processa as entradas e gera uma saída
	 * @param input Entradas vindas da camada anterior
	 */
	virtual void execute(const vec_float input) = 0;

	/**
	 * Atualiza os pesos das conexões e calcula os erros cometidos
	 * @param input Entradas vindas da camada anterior
	 * @param signal Sinal de feedback vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void response(const vec_float input, float signal, float learning)
		= 0;

	/**
	 * Retorna o peso com a i-ésima entrada
	 * @param i Índice da entrada
	 * @return Peso com a i-ésima entrada
	 */
	float getWeight(uint i);

	/**
	 * Seta o peso com a i-ésima entrada
	 * @param i Índice da entrada
	 * @param weight Peso com a i-ésima entrada
	 */
	void setWeight(uint i, float weight);

protected:

	/**
	 * Quantidade de entradas
	 */
	uint inUnits;

	/**
	 * Pesos das conexões com as entradas
	 */
	hv_float weights;

	/**
	 * Gradiente
	 */
	float gradient;

};

}

#endif
