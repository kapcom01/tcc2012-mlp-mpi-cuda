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
	 * @param min Valor mínimo
	 * @param max Valor máximo
	 */
	LearningRate(double initialValue, double min, double max);

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
	 * Ajusta a taxa de aprendizado em função do erro
	 * @param error Erros cometidos na camada de saída
	 * @param expectedOutput Saída esperada
	 * @param size Tamanho dos vetores
	 */
	void adjust(const double* error, const double* expectedOutput, uint size);

private:

	/**
	 * Valor mínimo
	 */
	double min;

	/**
	 * Valor máximo
	 */
	double max;

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
