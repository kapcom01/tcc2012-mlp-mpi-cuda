#ifndef REMOTEOUTLAYER_H_
#define REMOTEOUTLAYER_H_

#include "mlp/common/OutLayer.h"
#include <mpi.h>

using namespace MPI;

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP em várias CPUs
 */
class RemoteOutLayer : public OutLayer
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
	 * Randomiza os pesos de todas as conexões com a camada anterior
	 */
	virtual void randomize();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	virtual void feedforward(const float* input);

	/**
	 * Calcula o erro da rede
	 * @param target Saída esperada para a rede neural
	 */
	virtual void calculateError(const float* target);

	/**
	 * Realiza a operação de feedforward
	 * @param target Saída esperada para a rede neural
	 * @param learning Taxa de aprendizado
	 */
	virtual void feedbackward(const float* target, float learning);

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

};

}

#endif
