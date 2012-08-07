#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "mlp/layer/Layer.h"

namespace MLP
{

/**
 * Classe que representa uma camada de saída da rede MLP
 */
class OutputLayer: public Layer
{

public:

	/**
	 * Constrói uma camada escondida
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnit Número de neurônios na camada atual
	 * @param activation Função de ativação
	 */
	OutputLayer(uint inUnits, uint outUnits,
			const ActivationFunc* activation);

	/**
	 * Destrói a camada
	 */
	virtual ~OutputLayer();

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
