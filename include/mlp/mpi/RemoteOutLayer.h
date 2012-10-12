#ifndef REMOTEOUTLAYER_H_
#define REMOTEOUTLAYER_H_

#include "mlp/serial/HostOutLayer.h"
#include "mlp/mpi/RemoteUtils.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP em diversas CPUs
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
	void feedforward(const vec_float &input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	void feedback(const vec_float &signal, float learning);

protected:

	/**
	 * Constrói uma camada vazia
	 */
	RemoteOutLayer();

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

};

}

#endif
