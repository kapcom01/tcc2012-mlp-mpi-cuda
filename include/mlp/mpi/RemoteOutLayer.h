#ifndef REMOTEOUTLAYER_H_
#define REMOTEOUTLAYER_H_

#include "mlp/serial/HostOutLayer.h"
#include <mpi.h>

using namespace MPI;

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP em várias CPUs
 */
class RemoteOutLayer : public HostOutLayer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 * @param hid ID do host
	 * @param hosts Quantidade de hosts
	 */
	RemoteOutLayer(uint inUnits, uint outUnits, uint hid, uint hosts);

	/**
	 * Destrói a camada
	 */
	virtual ~RemoteOutLayer();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const float* input);

	/**
	 * Realiza a operação de feedforward
	 * @param target Saída esperada para a rede neural
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* target, float learning);

protected:

	/**
	 * ID do host
	 */
	uint hid;

};

}

#endif
