#ifndef REMOTELAYER_H_
#define REMOTELAYER_H_

#include "mlp/common/Layer.h"
#include "mlp/mpi/BalanceInfo.h"
#include <mpi.h>

using namespace MPI;

namespace ParallelMLP
{

/**
 * Classe que representa uma camada da rede MLP em diversas CPUs
 */
class RemoteLayer : public Layer
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
	 * Randomiza os pesos de todas as conexões com a camada anterior
	 */
	virtual void randomize();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const float* input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* signal, float learning);

protected:

	/**
	 * Retorna um valor aleatório entre -1 e 1
	 * @return Valor aleatório entre -1 e 1
	 */
	float random() const;

	/**
	 * ID do host
	 */
	uint hid;

	/**
	 * Informações do balanceamento
	 */
	BalanceInfo binfo;

	/**
	 * Quantidade de neurônios gerenciados por este nó
	 */
	uint toutUnits;

	/**
	 * Quantidade de conexões gerenciadas por este nó
	 */
	uint tconnUnits;

	/**
	 * Offset deste nó
	 */
	uint offset;

	/**
	 * Sinal funcional gerenciado por este nó
	 */
	float* tfuncSignal;

};

}

#endif
