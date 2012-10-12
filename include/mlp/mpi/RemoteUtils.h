#ifndef REMOTEUTILS_H_
#define REMOTEUTILS_H_

#include "mlp/Vector.h"

namespace ParallelMLP
{

/**
 * Classe que realiza algumas funções utéis para MPI
 */
class RemoteUtils
{

public:

	/**
	 * ID do nó mestre
	 */
	enum
	{
		MASTER = 0
	};

	/**
	 * Verifica se um host é mestre
	 * @param hid ID do host
	 * @return Verdadeiro se for mestre ou falso caso contrário
	 */
	static bool isMaster(uint hid);

	/**
	 * Verifica se um host é escravo
	 * @param hid ID do host
	 * @return Verdadeiro se for escravo ou falso caso contrário
	 */
	static bool isSlave(uint hid);

	/**
	 * Realiza o balanceamento de uma certa quantidade de itens
	 * @param total Quantidade total de itens
	 */
	static void balance(uint total, v_int &counts, v_int &offset);

};

}

#endif
