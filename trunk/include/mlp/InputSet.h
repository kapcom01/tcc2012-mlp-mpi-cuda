#ifndef INPUTSET_H_
#define INPUTSET_H_

#include "Common.h"

namespace MLP
{

/**
 * Classe que contém os dados de treinamento
 */
class InputSet
{

public:

	/**
	 * Constrói um conjunto de entrada
	 * @param size Quantidade de instâncias
	 * @param inVars Quantiadade de variáveis de entrada
	 * @param outVars Quantiadade de variáveis de saída
	 */
	InputSet(uint size, uint inVars, uint outVars);

	/**
	 * Destrói o conjunto de entradas
	 */
	virtual ~InputSet();

	/**
	 * Quantidade de variáveis de entrada
	 */
	uint inVars;

	/**
	 * Quantidade de variáveis de saída
	 */
	uint outVars;

	/**
	 * Quantidade de instâncias
	 */
	uint size;

	/**
	 * Dados de entrada do treinamento
	 */
	double** input;

	/**
	 * Dados de saída esperada para o treinamento
	 */
	double** expectedOutput;

	/**
	 * Dados de saída da rede neural
	 */
	double **output;

	/**
	 * Taxa de sucesso
	 */
	double successRate;

};

}

#endif
