#ifndef HOSTNEURON_H_
#define HOSTNEURON_H_

#include "mlp/common/Neuron.h"

namespace ParallelMLP
{

/**
 * Classe que representa um neurônio na CPU
 */
class HostNeuron : public Neuron
{

public:

	/**
	 * Constrói um neurônio
	 * @param inUnits Quantidade de entradas
	 * @param output Saída do neurônio
	 * @param error Erros causados pelo neurônio
	 */
	HostNeuron(uint inUnits, float &output, hv_float &error);

	/**
	 * Destrói o neurônio
	 */
	virtual ~HostNeuron();

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Processa as entradas e gera uma saída
	 * @param input Entradas vindas da camada anterior
	 */
	void execute(const vec_float input);

	/**
	 * Atualiza os pesos das conexões e calcula os erros cometidos
	 * @param input Entradas vindas da camada anterior
	 * @param signal Sinal de feedback vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	void response(const vec_float input, float signal, float learning);

private:

	/**
	 * Sinal funcional
	 */
	float &output;

	/**
	 * Retorno de erro
	 */
	hv_float &error;

};

}

#endif
