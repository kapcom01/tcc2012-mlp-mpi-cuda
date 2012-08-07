#ifndef ACTIVATIONFUNC_H_
#define ACTIVATIONFUNC_H_

#include "Common.h"
#include <cmath>
#include <ctime>

namespace MLP
{

/**
 * Classe que representa uma função de ativação
 */
class ActivationFunc
{

public:

	/**
	 * Contrói uma função de ativação
	 */
	ActivationFunc();

	/**
	 * Destrói a função de ativação
	 */
	virtual ~ActivationFunc();

	/**
	 * Calcula o valor da função de ativação para o valor passado
	 * @param x Valor que será ativado
	 * @return Valor ativado
	 */
	virtual double activate(double x) const = 0;

	/**
	 * Calcula o valor da derivada da função de ativação para o valor passado
	 * @param x Valor que será ativado
	 * @return Valor ativado
	 */
	virtual double derivate(double x) const = 0;

	/**
	 * Retorna o valor inicial de um neurônio
	 * @param inUnits Número de unidades de entrada
	 * @param outUnits Número de unidades de saída
	 * @return Valor inicial de um neurônio
	 */
	virtual double initialValue(uint inUnits, uint outUnits) const = 0;

protected:

	/**
	 * Retorna um valor aleatório entre min e max
	 * @param min Valor mínimo
	 * @param max Valor máximo
	 * @return Valor aleatório entre min e max
	 */
	double randomBetween(double min, double max) const;

};

/**
 * Ponteiro para ActivationFunction
 */
typedef shared_ptr<ActivationFunc> ActivationPtr;

}

#endif
