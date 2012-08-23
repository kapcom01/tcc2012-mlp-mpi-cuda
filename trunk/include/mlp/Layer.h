#ifndef LAYER_H_
#define LAYER_H_

#include "Common.h"
#include "mlp/activation/ActivationFunc.h"

#define MAX_INIT_WEIGHT 0.02

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
	Layer(uint inUnits, uint outUnits, const ActivationFunc &activation);

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
	void feedforward(const vector<double> &input);

	/**
	 * Realiza o feedforward dos neurônios
	 * @param signal Sinal vindo da camada posterior
	 * @param learningRate Taxa de aprendizado
	 * @param momentum Momento
	 */
	void feedback(const vector<double> &signal, double learningRate,
			double momentum);

	friend class Database::MLPHelper;
	friend class BackpropMLP;

private:

	/**
	 * Retorna um valor aleatório para um peso inicial
	 * @return Valor aleatório para um peso inicial
	 */
	double randomWeight() const;

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
	const ActivationFunc &activation;

	/**
	 * Matriz contendo os pesos de cada neurônio para cada entrada
	 */
	vector<vector<double>> weights;

	/**
	 * Matriz contendo as variações dos pesos
	 */
	vector<vector<double>> delta;

	/**
	 * Entrada vinda da camada anterior
	 */
	const vector<double>* input;

	/**
	 * Saída ativada dos neurônios (sinal funcional)
	 */
	vector<double> funcSignal;

	/**
	 * Sinal de erro
	 */
	vector<double> errorSignal;

	/**
	 * Vetor de gradiente
	 */
	vector<double> gradient;

};

/**
 * Ponteiro para Layer
 */
typedef shared_ptr<Layer> LayerPtr;

}

#endif
