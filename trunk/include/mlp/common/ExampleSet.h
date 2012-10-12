#ifndef EXAMPLESET_H_
#define EXAMPLESET_H_

#include "mlp/Vector.h"

namespace ParallelMLP
{

/**
 * Tipos de conjunto de dados
 */
enum SetType
{
	TRAINING = 1, VALIDATION = 2, TEST = 3
};

/**
 * Classe que contém um conjunto de dados experimentais
 */
class ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 * @param relationID ID da relação
	 * @param mlpID ID da rede
	 * @param type Tipo do conjunto de dados
	 */
	ExampleSet(int relationID, int mlpID, SetType type);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~ExampleSet();

	/**
	 * Normaliza as entradas e saídas alvo do conjunto de dados
	 */
	virtual void normalize() = 0;

	/**
	 * Desnormaliza as entradas, saídas alvo e saídas do conjunto de dados
	 */
	virtual void unnormalize() = 0;

	/**
	 * Adiciona um valor númerico de entrada ou saída
	 * @param value Valor numérico de entrada ou saída
	 * @param isTarget Indica se o valor é de saída
	 */
	void addValue(float value, bool isTarget);

	/**
	 * Adiciona um valor nominal de entrada ou saída
	 * @param value Valor nominal de entrada ou saída
	 * @param card Cardinalidade do atributo nominal
	 * @param isTarget Indica se o valor é de saída
	 */
	void addValue(int value, uint card, bool isTarget);

	/**
	 * Adiciona um valor estatístico numérico de entrada ou saída
	 * @param min Valor mínimo da amostra
	 * @param max Valor máximo da amostra
	 * @param lower Menor valor depois de normalizado
	 * @param upper Maior valor depois de normalizado
	 * @param isTarget Indica se o valor é de saída
	 */
	void addStat(float min, float max, float lower, float upper,
			bool isTarget);

	/**
	 * Adiciona um valor estatístico nominal de entrada ou saída
	 * @param lower Menor valor depois de normalizado
	 * @param upper Maior valor depois de normalizado
	 * @param card Cardinalidade do atributo nominal
	 * @param isTarget Indica se o valor é de saída
	 */
	void addStat(float lower, float upper, uint card, bool isTarget);

	/**
	 * Verifica se é um conjunto de treinamento
	 * @return Verdadeiro se for de treinamento ou falso caso contrário
	 */
	bool isTraining() const;

	/**
	 * Verifica se é um conjunto de validação
	 * @return Verdadeiro se for de validação ou falso caso contrário
	 */
	bool isValidation() const;

	/**
	 * Verifica se é um conjunto de teste
	 * @return Verdadeiro se for de teste ou falso caso contrário
	 */
	bool isTest() const;

	/**
	 * Retorna o ID do conjunto de dados
	 * @return ID do conjunto de dados
	 */
	int getID() const;

	/**
	 * Retrona o ID do MLP que utilizará esse conjunto de dados
	 * @return ID do MLP que utilizará esse conjunto de dados
	 */
	int getMLPID() const;

	/**
	 * Retorna a quantidade de variáveis de entrada
	 * @return Quantidade de variáveis de entrada
	 */
	uint getInVars() const;

	/**
	 * Retorna a quantidade de variáveis de saída
	 * @return Quantidade de variáveis de saída
	 */
	uint getOutVars() const;

	/**
	 * Finaliza a inserção de dados
	 */
	void done();

	/**
	 * Retorna o tamanho do conjunto de entrada
	 * @return Tamanho do conjunto de entrada
	 */
	uint getSize() const;

	/**
	 * Seta a quantidade de instâncias
	 * @param size Quantidade de instâncias
	 */
	void setSize(uint size);

	/**
	 * Retorna a i-ésima entrada do conjunto
	 * @param i Índice da entrada
	 * @return Entrada de índice i
	 */
	vec_float getInput(uint i);

	/**
	 * Retorna a i-ésima saída alvo do conjunto
	 * @param i Índice da saída alvo
	 * @return Saída alvo de índice i
	 */
	vec_float getTarget(uint i);

	/**
	 * Retorna a k-ésima saída numérica da rede neural
	 * @param k Índice da instância
	 * @return k-ésima saída numérica da rede neural
	 */
	float getNumericOutput(uint k) const;

	/**
	 * Retorna a k-ésima saída nominal da rede neural
	 * @param k Índice da instância
	 * @return k-ésima saída nominal da rede neural
	 */
	int getNominalOutput(uint k) const;

	/**
	 * Seta os valores da i-ésima saída
	 * @param output Vetor contendo a i-ésima saída
	 */
	void setOutput(uint i, vec_float &output);

	/**
	 * Retorna o tipo do conjunto de dados
	 * @return Tipo do conjunto de dados
	 */
	int getType() const;

	/**
	 * Retorna a taxa de aprendizado
	 * @return Taxa de aprendizado
	 */
	float getLearning() const;

	/**
	 * Seta a taxa de aprendizado
	 * @param learning Taxa de aprendizado
	 */
	void setLearning(float learning);

	/**
	 * Retorna a tolerância
	 * @return Tolerância
	 */
	float getTolerance() const;

	/**
	 * Seta a tolerância
	 * @param tolerance Tolerância
	 */
	void setTolerance(float tolerance);

	/**
	 * Retorna a quantidade máxima de épocas
	 * @return Quantidade máxima de épocas
	 */
	uint getMaxEpochs() const;

	/**
	 * Seta a quantidade máxima de épocas
	 * @param maxEpochs Quantidade máxima de épocas
	 */
	void setMaxEpochs(uint maxEpochs);

	/**
	 * Retorna o erro quadrático médio cometido
	 * @return Erro quadrático médio cometido
	 */
	float getError() const;

	/**
	 * Seta o erro quadrático médio cometido
	 * @param error Erro quadrático médio cometido
	 */
	void setError(float error);

	/**
	 * Retorna a quantidade de épocas utilizadas
	 * @return Quantidade de épocas utilizadas
	 */
	uint getEpochs() const;

	/**
	 * Seta a quantidade de épocas utilizadas
	 * @param epochs Quantidade de épocas utilizadas
	 */
	void setEpochs(uint epochs);

	/**
	 * Retorna o tempo da operação em milisegundos
	 * @return Tempo da operação em milisegundos
	 */
	float getTime() const;

	/**
	 * Seta o tempo da operação em milisegundos
	 * @param time Tempo da operação em milisegundos
	 */
	void setTime(float time);

	void print();

protected:

	/**
	 * ID da relação
	 */
	int relationID;

	/**
	 * ID da rede
	 */
	int mlpID;

	/**
	 * Tipo do conjunto de dados
	 */
	SetType type;

	/**
	 * Taxa de aprendizado
	 */
	float learning;

	/**
	 * Número máximo de épocas
	 */
	uint maxEpochs;

	/**
	 * Tolerância máxima
	 */
	float tolerance;

	/**
	 * Quantidade de variáveis de entrada
	 */
	uint inVars;

	/**
	 * Quantidade de variáveis de saída
	 */
	uint outVars;

	/**
	 * Quantidade de instâncias
	 */
	uint size;

	/**
	 * Dados de entrada do treinamento
	 */
	hv_float input;

	/**
	 * Dados de saída da rede neural
	 */
	hv_float output;

	/**
	 * Estatísticas para cada coluna de dados
	 */
	hv_stat stat;

	/**
	 * Erro cometido pela rede
	 */
	float error;

	/**
	 * Tempo de execução da operação
	 */
	float time;

	/**
	 * Quantidade de iterações feitas
	 */
	uint epochs;

	/**
	 * Indica se o conjunto de dados está normalizado
	 */
	bool isNormalized;

};

}

#endif
