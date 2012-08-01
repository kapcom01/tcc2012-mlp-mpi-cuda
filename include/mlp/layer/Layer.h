#ifndef LAYER_H_
#define LAYER_H_

#include "Common.h"
#include "mlp/activation/ActivationFunction.h"
#include "mlp/LearningRate.h"

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
	 * @param learningRate Taxa de aprendizado
	 */
	Layer(uint inUnits, uint outUnits, const ActivationFunction* activation,
			const LearningRate* learningRate);

	/**
	 * Destrói a camada
	 */
	virtual ~Layer();

	/**
	 * Retorna a saída ativada
	 * @return Saída ativada
	 */
	double* getOutput();

	/**
	 * Retorna o sinal de feedback
	 * @return Sinal de feedback
	 */
	double* getFeedback();

	/**
	 * Realiza o feedforward dos neurônios
	 * @param input Entrada vindo da camada anterior
	 */
	void feedforward(const double* input);

	/**
	 * Realiza o feedforward dos neurônios
	 * @param signal Sinal vindo da camada posterior
	 * @param output Vetor de valores reversamente ativados dos neurônios
	 */
	void feedback(const double* signal);

	/**
	 * Calcula o i-ésimo erro
	 * @param i Índice do erro a ser calculado
	 * @param signal Sinal vindo da camada posterior
	 * @return i-ésimo error
	 */
	virtual double calculateError(uint i, const double* signal) = 0;

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
	const ActivationFunction* activation;

	/**
	 * Matriz contendo os pesos de cada neurônio para cada entrada
	 */
	double** weights;

	/**
	 * Entrada vinda da camada anterior
	 */
	const double* input;

	/**
	 * Saída não ativada dos neurônios
	 */
	double* nonActivatedOutput;

	/**
	 * Saída ativada dos neurônios
	 */
	double* activatedOutput;

	/**
	 * Sinal de feedback
	 */
	double* feedbackSignal;

	/**
	 * Erros cometidos por cada neurônio
	 */
	double* error;

private:

	/**
	 * Taxa de aprendizado
	 */
	const LearningRate* learningRate;

};

/**
 * Ponteiro para Layer
 */
typedef shared_ptr<Layer> LayerPtr;

}

#endif
