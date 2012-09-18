#ifndef PARSEEXCEPTION_H_
#define PARSEEXCEPTION_H_

#include "Common.h"
#include "arff/Driver.h"
#include <unordered_map>

#define throwError(A) throw ParseException(A, driver)

namespace ParallelMLP
{

/**
 * Tipos de erros
 */
enum
{
	LEX_MALFORMED_NUMBER, // Nùmero mal formado
	LEX_QUOTATION_NOT_CLOSED, // Aspas não fechada
	LEX_UNKNOWN_TOKEN, // Token desconhecido

	SIN_INVALID_RELATION, // Declaração inválida de relação
	SIN_INVALID_ATTRIBUTE, // Declaração inválida de atributo
	SIN_INVALID_INSTANCE, // Declaração inválida de instância

	SEM_TYPE_NOT_ALLOWED, // Tipo de atributo não permitido
	SEM_WRONG_INSTANCE_TYPE, // Tipo de instância inválido
	SEM_SAME_ATTRIBUTE_NAME, // Nome de atributo já declarado
	SEM_SAME_NOMINAL_VALUE, // Valor nominal já declarado
	SEM_NOMINAL_NOT_DECLARED // Valor nominal não declarado

};

/**
 * Classe que contém informações de uma exceção causada durante o parseamento
 */
class ParseException: public exception
{

public:

	/**
	 * Constrói uma nova exceção
	 * @param error Tipo do erro
	 * @param driver Driver
	 */
	ParseException(int error, Driver &driver);

	/**
	 * Destrói a exceção
	 */
	virtual ~ParseException() throw ();

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
