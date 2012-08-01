#ifndef LOGISTICFUNCTION_H_
#define LOGISTICFUNCTION_H_

#include "mlp/activation/ActivationFunction.h"

namespace MLP
{

/**
 * Classe que representa uma função de ativação logística
 */
class LogisticFunction : public ActivationFunction
{

public:

	/**
	 * Contrói uma função de ativação logística
	 */
	LogisticFunction();

	/**
	 * Destrói a função de ativação
	 */
	virtual ~LogisticFunction();

	/**
	 * Calcula o valor da função de ativação para o valor passado
	 * @param x Valor que será ativado
	 * @return Valor ativado
	 */
	double activate(double x) const;

	/**
	 * Calcula o valor da derivada da função de ativação para o valor passado
	 * @param x Valor que será ativado
	 * @return Valor ativado
	 */
	double derivate(double x) const;

	/**
	 * Retorna o valor inicial de um neurônio
	 * @param inUnits Número de unidades de entrada
	 * @param outUnits Número de unidades de saída
	 * @return Valor inicial de um neurônio
	 */
	double initialValue(uint inUnits, uint outUnits) const;

};

}

#endif
