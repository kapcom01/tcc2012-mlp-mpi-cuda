#ifndef PARALLELMLPEXCEPTION_H_
#define PARALLELMLPEXCEPTION_H_

#include "Common.h"
#include <map>

namespace ParallelMLP
{

/**
 * Tipos de erros
 */
enum ErrorType
{
	// Erros de parseamento de arquivos ARFF
	LEX_MALFORMED_NUMBER,		// Número mal formado
	LEX_QUOTATION_NOT_CLOSED,	// Aspas não fechada
	LEX_UNKNOWN_TOKEN,			// Token desconhecido
	SIN_INVALID_RELATION,		// Declaração inválida de relação
	SIN_INVALID_ATTRIBUTE,		// Declaração inválida de atributo
	SIN_INVALID_INSTANCE,		// Declaração inválida de instância
	SEM_TYPE_NOT_ALLOWED,		// Tipo de atributo não permitido
	SEM_WRONG_INSTANCE_TYPE,	// Tipo de instância inválido
	SEM_SAME_ATTRIBUTE_NAME,	// Nome de atributo já declarado
	SEM_SAME_NOMINAL_VALUE,		// Valor nominal já declarado
	SEM_NOMINAL_NOT_DECLARED,	// Valor nominal não declarado

	// Erros de operações em bases de dados
	COULD_NOT_CONNECT,			// Não foi possível se conectar à base de dados
	RELATION_NOT_UNIQUE,		// Nome da relação já existe

	// Erros de operações de um MLP
	INVALID_INPUT_VARS,			// Quantidade de entradas inválida
	INVALID_OUTPUT_VARS,		// Quantidade de saídas inválida

	// Outros erros
	OTHER_ERRORS				// Outros erros
};

/**
 * Classe que contém informações de uma exceção
 */
class ParallelMLPException: public exception
{

public:

	/**
	 * Constrói uma nova exceção
	 * @param error Tipo do erro
	 */
	ParallelMLPException(ErrorType error);

	/**
	 * Constrói uma nova exceção
	 * @param error Tipo do erro
	 * @param info Informações da mensagem de erro
	 */
	ParallelMLPException(ErrorType error, string info);

	/**
	 * Constrói uma nova exceção
	 * @param error Tipo do erro
	 * @param info Informações da mensagem de erro
	 * @param lineno Número da linha
	 */
	ParallelMLPException(ErrorType error, string info, int lineno);

	/**
	 * Constrói a partir de uma outra exceção
	 * @param ex Exceção qualquer
	 */
	ParallelMLPException(const exception& ex);

	/**
	 * Destrói a exceção
	 */
	virtual ~ParallelMLPException() throw ();

	/**
	 * Retorna a mensagem do erro
	 * @return Mensagem de erro
	 */
	virtual const char* what() const throw ();

	/**
	 * Retorna o número do erro
	 * @return Número do erro
	 */
	ErrorType getErrorType() const;

private:

	/**
	 * Número do erro
	 */
	ErrorType error;

	/**
	 * Mensagem
	 */
	string msg;

	/**
	 * Todas as mensagens indexadas pelo erro
	 */
	static map<int, string> messages;

};

}

#endif
