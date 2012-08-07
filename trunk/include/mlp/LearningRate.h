#ifndef LEARNINGRATE_H_
#define LEARNINGRATE_H_

#include "Common.h"
#include <cmath>
#include <algorithm>

namespace MLP
{

/**
 * Classe que representa a taxa de aprendizado da rede neural
 */
class LearningRate
{

public:

	/**
	 * Constrói uma taxa de aprendizado
	 * @param initialValue Valor inicial
	 * @param searchTime Número de iterações para busca
	 */
	LearningRate(double initialValue, uint searchTime);

	/**
	 * Destrói a taxa de aprendizado
	 */
	virtual ~LearningRate();

	/**
	 * Retorna o valor da taxa de aprendizado
	 * @return Valor da taxa de aprendizado
	 */
	double get() const;

	/**
	 * Retorna o valor da taxa de aprendizado
	 * @return Valor da taxa de aprendizado
	 */
	double operator *() const;

	/**
	 * Ajusta a taxa de aprendizado em função da quantidade de iterações
	 * @param k Quantidade de iterações
	 */
	void adjust(uint iteration);

private:

	/**
	 * Taxa de aprendizado inicial
	 */
	double initial;

	/**
	 * Número de iterações para busca
	 */
	uint searchTime;

	/**
	 * Taxa de aprendizado
	 */
	double learningRate;

};

/**
 * Ponteiro para LearningRate
 */
typedef shared_ptr<LearningRate> LearningRatePtr;

}

#endif
