#ifndef TYPES_H_
#define TYPES_H_

#include "Common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace thrust;

namespace ParallelMLP
{

/**
 * Vetor de double
 */
typedef host_vector<double> vdouble;

/**
 * Vetor de uint
 */
typedef host_vector<uint> vuint;

/**
 * Vetor de double no dispositivo
 */
typedef device_vector<double> dvdouble;

}

#endif
