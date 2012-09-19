#ifndef LAYER_H_
#define LAYER_H_

#include "mlp/serial/Neuron.h"

namespace ParallelMLP
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
	 */
	Layer(uint inUnits, uint outUnits);

	/**
	 * Destrói a camada
	 */
	virtual ~Layer();

	/**
	 * Randomiza os pesos de todas as conexões com a camada anterior
	 */
	void randomize();

	/**
	 * Realiza a operação de feedforward
	 * @param input Sinal funcional vindo da camada anterior
	 */
	void feedforward(const hv_float &input);

	/**
	 * Realiza a operação de feedforward
	 * @param signal Sinal de erro vindo da camada posterior
	 * @param learning Taxa de aprendizado
	 */
	void feedback(const hv_float &signal, float learning);

	friend class BackpropMLPAdapter;
	friend class BackpropMLP;

private:

	/**
	 * Número de neurônios na camada anterior
	 */
	uint inUnits;

	/**
	 * Número de neurônios na camada atual
	 */
	uint outUnits;

	/**
	 * Neurônios da camada
	 */
	vector<NeuronPtr> neurons;

	/**
	 * Entrada vinda da camada anterior
	 */
	const hv_float* input;

	/**
	 * Sinal funcional dos neurônios
	 */
	hv_float funcSignal;

	/**
	 * Sinal de erro
	 */
	hv_float errorSignal;

};

/**
 * Ponteiro para Layer
 */
typedef shared_ptr<Layer> LayerPtr;

}

#endif
