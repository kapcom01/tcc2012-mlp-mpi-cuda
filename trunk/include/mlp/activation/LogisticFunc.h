#ifndef LOGISTICFUNC_H_
#define LOGISTICFUNC_H_

#include "mlp/activation/ActivationFunc.h"

namespace MLP
{

/**
 * Classe que representa uma função de ativação logística
 */
class LogisticFunc : public ActivationFunc
{

public:

	/**
	 * Contrói uma função de ativação logística
	 */
	LogisticFunc();

	/**
	 * Destrói a função de ativação
	 */
	virtual ~LogisticFunc();

	/**
	 * Calcula o valor da função de ativação para o valor passado
	 * @param x Valor que será ativado
	 * @return Valor ativado
	 */
	double activate(double x) const;

	/**
	 * Calcula o valor da derivada da função de ativação para o valor passado
	 * @param y Valor ativado
	 * @return Valor "desativado"
	 */
	double derivate(double y) const;

	/**
	 * Retorna o intervalo de valores que a função pode retornar
	 * @return Intervalo de valores que a função pode retornar
	 */
	Range getRange() const;

};

}

#endif
