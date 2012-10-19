#ifndef TYPES_H_
#define TYPES_H_

#include "Common.h"
#include <curand_kernel.h>

namespace ParallelMLP
{

/**
 * Estrutura que armazena um intervalo
 */
struct Range
{
	/**
	 * Valor mínimo
	 */
	float lower;

	/**
	 * Valor máximo
	 */
	float upper;
};

/**
 * Estrutura contendo estatísticas sobre o conjunto de entrada
 */
struct Stat
{
	/**
	 * Intervalo de valores de um dado
	 */
	Range from;

	/**
	 * Intervalo de valores de um dado normalizado
	 */
	Range to;
};

/**
 * Vetor de floats
 */
typedef vector<float> v_float;

/**
 * Vetor de inteiros
 */
typedef vector<int> v_int;

/**
 * Vetor de inteiros sem sinal
 */
typedef vector<int> v_uint;

/**
 * Vetor de estatísticas
 */
typedef vector<Stat> v_stat;

}

#endif
