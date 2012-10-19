#ifndef INDEXES_H_
#define INDEXES_H_

#include <mlp/Types.h>

namespace ParallelMLP
{

/**
 * Classe que contém índices
 */
class Indexes
{

public:

    /**
     * Constrói um vetor de índices
     */
	Indexes();

	/**
	 * Seta a quantidade de índices
	 * @param size Quantidade de índices
	 */
	void resize(uint size);

	/**
	 * Randomiza os índices
	 */
	void randomize();

	/**
	 * Retorna um índice aleatório
	 * @param i Posição do índice
	 */
	uint get(uint i) const;

private:

    /**
     * Vetor de índices
     */
	v_uint indexes;

};

}

#endif
