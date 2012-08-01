#ifndef SETTINGS_H_
#define SETTINGS_H_

#include "Common.h"

namespace MLP
{

/**
 * Tipos de função de ativação
 */
enum ActivationType
{
	HYPERBOLIC, LOGISTIC
};

/**
 * Tipos de treinamento
 */
enum TrainingType
{
	CLASSIFICATION, APROXIMATION
};

/**
 * Classe que contém as configurações da rede MLP
 */
class Settings
{

public:

	/**
	 * Constrói estrutura para armazenar as configurações de uma rede MLP
	 * @param nHiddenLayers Quantidade de camadas escondidas
	 */
	Settings(uint nLayers);

	/**
	 * Destrói as configurações
	 */
	virtual ~Settings();

	/**
	 * Quantidade de camadas
	 */
	uint nLayers;

	/**
	 * Número de neurônios em cada camada
	 */
	uint* units;

	/**
	 * Tolerância máxima
	 */
	double maxTolerance;

	/**
	 * Taxa de sucesso mínima
	 */
	double minSuccessRate;

	/**
	 * Taxa de aprendizado initial
	 */
	double initialLR;

	/**
	 * Taxa de aprendizado mínima
	 */
	double minLR;

	/**
	 * Taxa de aprendizado máxima
	 */
	double maxLR;

	/**
	 * Tipo da função de ativação
	 */
	ActivationType activationType;

	/**
	 * Tipo de treinamento
	 */
	TrainingType trainingType;

};

}

#endif
