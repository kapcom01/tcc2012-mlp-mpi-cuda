#ifndef TYPES_H_
#define TYPES_H_

#include "Common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

using namespace thrust;

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
 * Vetor de floats na CPU
 */
typedef host_vector<float> hv_float;

/**
 * Vetor de floats na GPU
 */
typedef device_vector<float> dv_float;

/**
 * Vetor de estatísticas na CPU
 */
typedef host_vector<Stat> hv_stat;

/**
 * Vetor de estatísticas na GPU
 */
typedef device_vector<Stat> dv_stat;

/**
 * Vetor de estados para geração de números aleatórios
 */
typedef device_vector<curandState> dv_rand;

}

#endif
