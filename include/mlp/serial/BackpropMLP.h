#ifndef BACKPROPMLP_H_
#define BACKPROPMLP_H_

#include "mlp/serial/Layer.h"
#include "mlp/serial/HostExampleSet.h"
#include <chrono>

using namespace chrono;

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron
 */
class BackpropMLP
{

public:

	/**
	 * Constrói um MLP que será recuperado
	 * @param mlpID ID da rede
	 */
	BackpropMLP(int mlpID);

	/**
	 * Constrói um MLP não treinado
	 * @param name Nome da rede
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	BackpropMLP(string name, vector<uint> &units);

	/**
	 * Destrói o MLP
	 */
	~BackpropMLP();

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Retorna o intervalo de valores de saída
	 * @return Intervalo de valores de saída
	 */
	Range getRange() const;

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	void train(HostExampleSet &training);

	/**
	 * Valida a rede neural
	 * @param validation Conjunto de validação
	 */
	void validate(HostExampleSet &validation);

	/**
	 * Testa a rede neural
	 * @param test Conjunto de testes
	 */
	void test(HostExampleSet &test);

	friend class BackpropMLPAdapter;

private:

	/**
	 * Copia a saída atual para o conjunto de dados
	 * @param set Conjunto de dados
	 * @param i Índice da entrada
	 */
	void copyOutput(HostExampleSet &set, uint i);

	/**
	 * Calcula o erro cometido pela rede
	 * @param target Saída alvo
	 */
	void calculateError(const hv_float &target);

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 */
	void feedforward(const hv_float &input);

	/**
	 * Realiza o feedback
	 * @param learning Taxa de aprendizado
	 */
	void feedback(float learning);

	/**
	 * Inicializa os índices
	 * @param indexes Vetor contendo os índices
	 */
	void initIndexes(vector<uint> &indexes) const;

	/**
	 * Embaralha os índices utilizando o algoritmo de Fisher-Yates
	 * @param indexes Vetor contendo os índices
	 */
	void shuffleIndexes(vector<uint> &indexes) const;

	/**
	 * ID da rede
	 */
	int mlpID;

	/**
	 * Nome da rede
	 */
	string name;

	/**
	 * Intervalo de valores para a saída
	 */
	Range range;

	/**
	 * Camadas
	 */
	vector<LayerPtr> layers;

	/**
	 * Saída da rede
	 */
	const hv_float* output;

	/**
	 * Erro cometido pela rede
	 */
	hv_float error;

	/**
	 * Erro total
	 */
	float totalError;

};

}

#endif
