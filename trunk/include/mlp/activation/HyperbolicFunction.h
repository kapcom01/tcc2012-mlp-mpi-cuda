#ifndef HYPERBOLICFUNCTION_H_
#define HYPERBOLICFUNCTION_H_

#include "mlp/activation/ActivationFunction.h"

namespace MLP
{

/**
 * Classe que representa uma função de ativação hiperbólica
 */
class HyperbolicFunction: public ActivationFunction
{

public:

	/**
	 * Contrói uma função de ativação hiperbólica
	 */
	HyperbolicFunction();

	/**
	 * Destrói a função de ativação
	 */
	virtual ~HyperbolicFunction();

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
