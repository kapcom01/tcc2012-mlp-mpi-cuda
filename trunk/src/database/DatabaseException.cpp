#include "database/DatabaseException.h"

namespace Database
{

/**
 * Inicializa as mensagens de erro
 */
unordered_map<int, string> DatabaseException::messages =
{
        { COULD_NOT_CONNECT,   "could not connect to database" },
        { RELATION_NOT_UNIQUE, "relation name already exists, choose another" }
};

/**
 * Constrói uma nova exceção
 * @param error Tipo do erro
 */
DatabaseException::DatabaseException(int error)
{
    msg = "ERROR:  " + messages[error];
}

/**
 * Constrói uma nova exceção
 * @param error Exceção dada pela biblioteca Pqxx
 */
DatabaseException::DatabaseException(pqxx_exception &error)
{
    msg = error.base().what();
}

/**
 * Retorna a mensagem do erro
 * @return Mensagem de erro
 */
const char* DatabaseException::what() const throw ()
{
    return msg.c_str();
}

}
