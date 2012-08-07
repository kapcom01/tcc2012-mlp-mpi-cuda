#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "Common.h"
#include <pqxx/pqxx>

using namespace pqxx;

/**
 * Ponteiro para work
 */
typedef shared_ptr<work> WorkPtr;

namespace Database
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
	 * Destrói a conexão
	 */
	virtual ~Connection();

	/**
	 * Retorna a conexão
	 * @return Conexão
	 */
	connection* get();

	/**
	 * Retorna um trabalho
	 * @return Trabalho
	 */
	WorkPtr getWork() const;

private:

	/**
	 * Conexão com a base de dados
	 */
	connection* conn;

};

}

#endif
