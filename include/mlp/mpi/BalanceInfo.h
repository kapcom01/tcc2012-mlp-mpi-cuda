#ifndef BALANCEINFO_H_
#define BALANCEINFO_H_

#include "mlp/Vector.h"

namespace ParallelMLP
{

/**
 * Classe que armazena informações sobre balanceamento
 */
class BalanceInfo
{

public:

	/**
	 * Constrói informações de balanceamento
	 */
	BalanceInfo();

	/**
	 * Destrói as informações do balanceamento
	 */
	~BalanceInfo();

	/**
	 * Altera o tamanho
	 * @param hosts Quantidade de hosts
	 */
	void resize(uint hosts);

	/**
	 * Balanceia um total entre os nós
	 * @param total Valor total
	 */
	void balance(uint total);

	/**
	 * Retorna a quantidade de valores em cada nó
	 * @return Quantidade de valores em cada nó
	 */
	int* getCounts();

	/**
	 * Retorna a quantidade de valores do nó
	 * @param hid ID do nó
	 */
	int getCount(uint hid);

	/**
	 * Retorna o offset para cada nó
	 * @return Offset para cada nó
	 */
	int* getOffsets();

	/**
	 * Retorna o offset do nó
	 * @param hid ID do nó
	 */
	int getOffset(uint hid);

protected:

	/**
	 * Quantidade de valores em cada nó
	 */
	v_int counts;

	/**
	 * Offset para cada nó
	 */
	v_int offsets;

};

}

#endif
