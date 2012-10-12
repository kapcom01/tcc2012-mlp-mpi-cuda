#ifndef OUTLAYER_H_
#define OUTLAYER_H_

#include "mlp/common/Layer.h"

namespace ParallelMLP
{

/**
 * Classe que representa uma camada de saída da rede MLP
 */
class OutLayer : virtual public Layer
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
	virtual void calculateError(const vec_float &target) = 0;

	/**
	 * Limpa o erro quadrático médio
	 */
	void clearTotalError();

	/**
	 * Incrementa o erro quadrático médio
	 * @param value Valor do incremento
	 * @param weight Peso do incremento
	 */
	void incTotalError(float value, uint weight = 1);

	/**
	 * Retorna o erro quadrático médio
	 * @return Erro quadrático médio
	 */
	float getTotalError();

protected:

	/**
	 * Constrói uma camada vazia
	 */
	OutLayer();

	/**
	 * Inicializa uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 */
	void init(uint inUnits, uint outUnits);

	/**
	 * Erro cometido pela rede
	 */
	vec_float rawError;

	/**
	 * Erro quadrático médio cometido pela rede
	 */
	float totalError;

	/**
	 * Quantidade de amostras para o erro total
	 */
	uint samples;

};

}

#endif
