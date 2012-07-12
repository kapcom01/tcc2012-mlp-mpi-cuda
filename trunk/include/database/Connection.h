#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "database/DatabaseException.h"

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
     * Retorna um trabalho
     * @return Trabalho
     */
    WorkPtr getWork();

private:

    /**
     * Conexão com a base de dados
     */
    connection* conn;

};

}

#endif
