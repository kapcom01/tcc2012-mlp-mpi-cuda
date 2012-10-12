#ifndef REMOTELAYER_H_
#define REMOTELAYER_H_

#include "mlp/serial/HostLayer.h"
#include "mlp/mpi/RemoteUtils.h"

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
	 * Quantidade de neurônios por host
	 */
	v_int counts;

	/**
	 * Offset relativo para cada host
	 */
	v_int offset;

};

}

#endif
