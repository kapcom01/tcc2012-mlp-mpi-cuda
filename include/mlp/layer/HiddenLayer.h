#ifndef HIDDENLAYER_H_
#define HIDDENLAYER_H_

#include "mlp/layer/Layer.h"

namespace MLP
{

/**
 * Classe que representa uma camada escondida da rede MLP
 */
class HiddenLayer: public Layer
{

public:

	/**
	 * Constrói uma camada escondida
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnit Número de neurônios na camada atual
	 * @param activation Função de ativação
	 */
	HiddenLayer(uint inUnits, uint outUnits,
			const ActivationFunc* activation);

	/**
	 * Destrói a camada
	 */
	virtual ~HiddenLayer();

	/**
	 * Calcula o i-ésimo erro
	 * @param i Índice do erro a ser calculado
	 * @param signal Sinal vindo da camada posterior
	 * @return i-ésimo error
	 */
	double calculateError(uint i, const double* signal);

};

}

#endif
