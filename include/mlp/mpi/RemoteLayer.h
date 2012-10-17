#ifndef REMOTELAYER_H_
#define REMOTELAYER_H_

#include "mlp/serial/HostLayer.h"
#include "mlp/mpi/BalanceInfo.h"
#include <mpi.h>

using namespace MPI;

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP em diversas CPUs
 */
class RemoteLayer : public HostLayer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 * @param hid ID do host
	 * @param hosts Quantidade de hosts
	 */
	RemoteLayer(uint inUnits, uint outUnits, uint hid, uint hosts);

	/**
	 * Destrói a camada
	 */
	virtual ~RemoteLayer();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const vec_float &input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedback(const vec_float &signal, float learning);

protected:

	/**
	 * Constrói uma camada vazia
	 */
	RemoteLayer();

	/**
	 * Inicializa uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 * @param hid ID do host
	 * @param hosts Quantidade de hosts
	 */
	void init(uint inUnits, uint outUnits, uint hid, uint hosts);

	/**
	 * ID do host
	 */
	uint hid;

	/**
	 * Informações do balanceamento
	 */
	BalanceInfo binfo;

};

}

#endif
