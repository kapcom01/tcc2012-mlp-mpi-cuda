#ifndef ACTIVATIONFUNC_H_
#define ACTIVATIONFUNC_H_

#include "Common.h"
#include <cmath>

namespace MLP
{

/**
 * Tipos de função de ativação
 */
enum ActivationType
{
	HYPERBOLIC = 1, LOGISTIC = 2
};

/**
 * Estrutura que armazena um intervalo
 */
struct Range
{
	/**
	 * Valor mínimo
	 */
	double min;

	/**
	 * Valor máximo
	 */
	double max;
};


/**
 * Classe que representa uma função de ativação
 */
class ActivationFunc
{

public:

	/**
	 * Contrói uma função de ativação
	 */
	ActivationFunc(int type);

	/**
	 * Destrói a função de ativação
	 */
	virtual ~ActivationFunc();

	/**
	 * Retorna o tipo da função de ativação
	 * @return Tipo da função de ativação
	 */
	int getType();

	/**
	 * Calcula o valor da função de ativação para o valor passado
	 * @param x Valor que será ativado
	 * @return Valor ativado
	 */
	virtual double activate(double x) const = 0;

	/**
	 * Calcula o valor da derivada da função de ativação para o valor passado
	 * @param y Valor ativado
	 * @return Valor "desativado"
	 */
	virtual double derivate(double y) const = 0;

	/**
	 * Retorna o intervalo de valores que a função pode retornar
	 * @return Intervalo de valores que a função pode retornar
	 */
	virtual Range getRange() const = 0;

protected:

	/**
	 * Tipo da função de ativação
	 */
	int type;

};

/**
 * Ponteiro para ActivationFunction
 */
typedef shared_ptr<ActivationFunc> ActivationFuncPtr;

}

#endif
