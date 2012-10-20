#ifndef OUTLAYER_H_
#define OUTLAYER_H_

#include "mlp/common/Layer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP
 */
class OutLayer : public virtual Layer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	OutLayer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~OutLayer();

	/**
	 * Calcula o erro da rede
	 * @param target Saída esperada para a rede neural
	 */
	virtual void calculateError(const float* target) = 0;

	/**
	 * Limpa o erro quadrático médio
	 */
	void clearError();

	/**
	 * Retorna o erro quadrático médio
	 * @return Erro quadrático médio
	 */
	float getError();

protected:

	/**
	 * Incrementa o erro em um certo valor
	 * @param inc Valor do incremento
	 */
	void incError(float inc);

	/**
	 * Vetor de erros
	 */
	float* error;

	/**
	 * Erro quadrático médio
	 */
	float totalError;

	/**
	 * Quantidade de amostras para o erro
	 */
	uint samples;

};

}

#endif
