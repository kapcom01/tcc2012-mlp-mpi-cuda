#ifndef LAYER_H_
#define LAYER_H_

#include "Common.h"
#include "mlp/activation/ActivationFunc.h"
#include "mlp/LearningRate.h"

namespace Database { class MLPHelper; }

namespace MLP
{

/**
 * Classe que representa uma camada da rede MLP
 */
class Layer
{

public:

	/**
	 * Constrói uma camada
	 * @param inUnits Número de neurônios na camada anterior
	 * @param outUnits Número de neurônios na camada atual
	 * @param activation Função de ativação
	 */
	Layer(uint inUnits, uint outUnits, const ActivationFunc* activation);

	/**
	 * Destrói a camada
	 */
	virtual ~Layer();

	/**
	 * Randomiza os pesos
	 */
	void randomizeWeights();

	/**
	 * Realiza o feedforward dos neurônios
	 * @param input Entrada vindo da camada anterior
	 */
	void feedforward(const double* input);

	/**
	 * Realiza o feedforward dos neurônios
	 * @param signal Sinal vindo da camada posterior
	 * @param learningRate Taxa de aprendizado
	 */
	void feedback(const double* signal, double learningRate);

	/**
	 * Calcula o i-ésimo erro
	 * @param i Índice do erro a ser calculado
	 * @param signal Sinal vindo da camada posterior
	 * @return i-ésimo error
	 */
	virtual double calculateError(uint i, const double* signal) = 0;

	friend class Database::MLPHelper;
	friend class BackpropMLP;

protected:

	/**
	 * Número de neurônios na camada anterior
	 */
	uint inUnits;

	/**
	 * Número de neurônios na camada atual
	 */
	uint outUnits;

	/**
	 * Função de ativação
	 */
	const ActivationFunc* activation;

	/**
	 * Matriz contendo os pesos de cada neurônio para cada entrada
	 */
	double** weights;

	/**
	 * Entrada vinda da camada anterior
	 */
	const double* input;

	/**
	 * Soma ponderada das entradas
	 */
	double* weightedSum;

	/**
	 * Saída ativada dos neurônios
	 */
	double* output;

	/**
	 * Sinal de feedback
	 */
	double* feedbackSignal;

	/**
	 * Erros cometidos por cada neurônio
	 */
	double* error;

};

/**
 * Ponteiro para Layer
 */
typedef shared_ptr<Layer> LayerPtr;

}

#endif
