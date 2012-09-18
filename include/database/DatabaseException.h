#ifndef DATABASEEXCEPTION_H_
#define DATABASEEXCEPTION_H_

#include "Common.h"
#include <pqxx/pqxx>
#include <unordered_map>

using pqxx::pqxx_exception;

namespace ParallelMLP
{

/**
 * Tipos de erros
 */
enum
{
	COULD_NOT_CONNECT, // Não foi possível se conectar à base de dados
	RELATION_NOT_UNIQUE // Nome da relação já existe
};

/**
 * Classe que contém informações de uma exceção na base de dados
 */
class DatabaseException: public exception
{

public:

	/**
	 * Constrói uma nova exceção
	 * @param error Tipo do erro
	 */
	DatabaseException(int error);

	/**
	 * Constrói uma nova exceção
	 * @param error Exceção dada pela biblioteca Pqxx
	 */
	DatabaseException(pqxx_exception &error);

	/**
	 * Destrói a exceção
	 */
	virtual ~DatabaseException() throw ();

	/**
	 * Retorna a mensagem do erro
	 * @return Mensagem de erro
	 */
	virtual const char* what() const throw ();

private:

	/**
	 * Mensagem
	 */
	string msg;

	/**
	 * Todas as mensagens indexadas pelo erro
	 */
	static unordered_map<int, string> messages;

};

}

#endif
