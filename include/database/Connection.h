#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "Common.h"
#include "exception/ParallelMLPException.h"
#include <pqxx/pqxx>

using namespace pqxx;

namespace ParallelMLP
{

/**
 * Classe que conecta na base de dados
 */
class Connection
{

public:

	/**
	 * Constrói uma nova conexão com a base de dados
	 */
	Connection();

	/**
	 * Destrói a conexão com a base de dados
	 */
	virtual ~Connection();

	/**
	 * Retorna a conexão
	 * @return Conexão
	 */
	static connection& get();

private:

	/**
	 * Conexão com a base de dados
	 */
	connection* baseConn;

	/**
	 * Conexão estática
	 */
	static shared_ptr<Connection> conn;

};

/**
 * Ponteiro para uma conexão
 */
typedef shared_ptr<Connection> ConnectionPtr;

}

#endif
