#ifndef MLP_H_
#define MLP_H_

#include "mlp/common/OutLayer.h"
#include "mlp/common/ExampleSet.h"
#include "mlp/common/Chronometer.h"

namespace ParallelMLP
{

/**
 * Classe que representa um Multi-Layer Perceptron
 */
class MLP
{

public:

	/**
	 * Constrói um MLP que será recuperado
	 * @param mlpID ID da rede
	 */
	MLP(int mlpID);

	/**
	 * Constrói um MLP não treinado
	 * @param name Nome da rede
	 * @param units Vetor contendo a quantidade de neurônios por camada
	 */
	MLP(string name, v_uint &units);

	/**
	 * Destrói o MLP
	 */
	virtual ~MLP();

	/**
	 * Realiza algumas configurações
	 */
	void config();

	/**
	 * Adiciona uma nova camada
	 * @param inUnits Unidades de entrada
	 * @param outUnits Unidades de saída
	 * @param isOutput Indica se é uma camada de saída
	 */
	virtual void addLayer(uint inUnits, uint outUnits, bool isOutput) = 0;

	/**
	 * Randomiza os pesos das conexões
	 */
	void randomize();

	/**
	 * Retorna o ID da rede
	 * @return ID da rede
	 */
	int getID() const;

	/**
	 * Seta o ID da rede
	 * @param mlpID ID da rede
	 */
	void setID(int mlpID);

	/**
	 * Retorna o nome da rede
	 * @return Nome da rede
	 */
	string getName() const;

	/**
	 * Seta o nome da rede
	 * @param name Nome da rede
	 */
	void setName(string name);

	/**
	 * Retorna o intervalo de valores de saída
	 * @return Intervalo de valores de saída
	 */
	Range getRange() const;

	/**
	 * Seta o intervalo de valores de saída
	 * @param range Intervalo de valores de saída
	 */
	void setRange(Range range);

	/**
	 * Retorna a quantidade de camadas
	 * @return Quantidade de camadas
	 */
	uint getNLayers() const;

	/**
	 * Retorna a i-ésima camada
	 * @param i Índice da camada
	 * @return i-ésima camada
	 */
	Layer& getLayer(uint i);

	/**
	 * Retorna a i-ésima camada
	 * @param i Índice da camada
	 * @return i-ésima camada
	 */
	const Layer& getLayer(uint i) const;

protected:

	/**
	 * Inicializa uma operação
	 * @param set Conjunto de dados
	 */
	void initOperation(ExampleSet* set);

	/**
	 * Finaliza uma operação
	 * @param set Conjunto de dados
	 */
	void endOperation(ExampleSet* set);

	/**
	 * Treina a rede neural
	 * @param training Conjunto de treinamento
	 */
	virtual void train(ExampleSet* training);

	/**
	 * Valida a rede neural
	 * @param validation Conjunto de validação
	 */
	virtual void validate(ExampleSet* validation);

	/**
	 * Testa a rede neural
	 * @param test Conjunto de testes
	 */
	virtual void test(ExampleSet* test);

	/**
	 * Realiza o feedforward
	 * @param input Dados de entrada
	 */
	void feedforward(const vec_float &input);

	/**
	 * Realiza o feedback
	 * @param learning Taxa de aprendizado
	 */
	void feedback(const vec_float &target, float learning);

	/**
	 * Inicializa os índices
	 * @param size Tamanho do vetor
	 */
	void initIndexes(uint size);

	/**
	 * Embaralha os índices utilizando o algoritmo de Fisher-Yates
	 */
	void shuffleIndexes();

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
	 * Vetor de índices para o treinamento
	 */
	v_uint indexes;

	/**
	 * Camadas
	 */
	vector<Layer*> layers;

	/**
	 * Ponteiro para a primeira camada
	 */
	Layer* firstLayer;

	/**
	 * Ponteiro para a última camada
	 */
	OutLayer* lastLayer;

	/**
	 * Cronômetro
	 */
	Chronometer chrono;

	/**
	 * Época atual
	 */
	uint epoch;

};

}

#endif
