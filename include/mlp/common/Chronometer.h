#ifndef CHRONOMETER_H_
#define CHRONOMETER_H_

#include <time.h>

namespace ParallelMLP
{

/**
 * Classe cuja função é cronometrar uma operação
 */
class Chronometer
{

public:

    /**
     * Constrói um cronômetro, inicializando o contador
     */
	Chronometer();

	/**
	 * Reseta o cronômetro
	 */
	void reset();

    /**
     * Retorna o tempo atual em segundos
     * @return Tempo atual em segundos
     */
	double getSeconds();

    /**
     * Retorna o tempo atual em milisegundos
     * @return Tempo atual em milisegundos
     */
	double getMiliseconds();

private:

    /**
     * Estrutura utilizada pelo cronômetro
     */
	struct timespec init;

};

}

#endif
