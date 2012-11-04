#ifndef CUDATYPES_H_
#define CUDATYPES_H_

#include "mlp/Types.h"
#include <curand.h>
#include <cublas_v2.h>

#define TPB 256

namespace ParallelMLP
{

/**
 * Classe de utilidades para CUDA
 */
class DeviceUtil
{

public:

	/**
	 * Manuseador da biblioteca CUBLAS
	 */
	static cublasHandle_t cublas;

};

}



#endif
