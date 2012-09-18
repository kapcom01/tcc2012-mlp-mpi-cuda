#ifndef NEURON_H_
#define NEURON_H_

#include "mlp/Types.h"
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
	 * @param output Saída do neurônio
	 * @param error Erros causados pelo neurônio
	 */
	Neuron(uint inUnits, double &output, vdouble &error);

	/**
	 * Destrói o neurônio
	 */
	virtual ~Neuron();

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Processa as entradas e gera uma saída
	 * @param input Entradas vindas da camada anterior
	 */
	void execute(const vdouble &input);

	/**
	 * Atualiza os pesos das conexões e calcula os erros cometidos
	 * @param input Entradas vindas da camada anterior
	 * @param signal Sinal de feedback vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	void response(const vdouble &input, double signal, double learning);

	friend class BackpropMLPAdapter;
	friend class BackpropMLP;

private:

	/**
	 * Retorna um valor aleatório entre -1 e 1
	 * @return Valor aleatório entre -1 e 1
	 */
	double random() const;

	/**
	 * Ativa um valor através da função de ativação hiperbólica
	 * @param x Soma das entradas ponderadas pelos pesos
	 * @return Sinal funcional
	 */
	double activate(double x) const;

	/**
	 * "Desativa" um valor através da derivada da função de ativação
	 * @param y Sinal de erro vindo da camada posterior
	 * @return Sinal "desativado"
	 */
	double derivate(double y) const;

	/**
	 * Quantidade de entradas
	 */
	uint inUnits;

	/**
	 * Sinal funcional
	 */
	double &output;

	/**
	 * Retorno de erro
	 */
	vdouble &error;

	/**
	 * Pesos das conexões com as entradas
	 */
	vdouble weights;

	/**
	 * Gradiente
	 */
	double gradient;

};

/**
 * Ponteiro para Neuron
 */
typedef shared_ptr<Neuron> NeuronPtr;

}

#endif
