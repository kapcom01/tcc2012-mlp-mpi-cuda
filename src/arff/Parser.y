%require "2.4.1"
%skeleton "lalr1.cc"

%defines
%define namespace "ParallelMLP"
%define parser_class_name "Parser"

%locations
%initial-action
{
	@$.begin.filename = @$.end.filename = &driver.filename;
};

%parse-param { Driver &driver }
%lex-param   { Driver &driver }

%error-verbose

%code requires
{
#include "arff/Relation.h"
#include "arff/Driver.h"
#include "exception/ParallelMLPException.h"
#define saveLine markLine(driver);
using namespace ParallelMLP;
}

%code provides
{
  /**
   * Chama o scanner para recuperar próximo token
   * @param yylval Valor semântico
   * @param yylloc Localização no arquivo
   * @param driver Driver
   */
  int yylex(Parser::semantic_type* yylval, Parser::location_type* yyloc,
		  Driver &driver);

  /**
   * Marca a linha atual
   * @param driver Driver
   */
  void markLine(Driver &driver);
  
  /**
   * Lança uma exceção
   * @param error Tipo do erro
   * @param driver Driver
   */
  void throwError(ErrorType error, Driver &driver);
}

/**
 * Tipo dos tokens
 */
%union
{
	string* sval;
	int ival;
	float fval;
	Nominal* nominal;
	Value* value;
	DataList* row;
}

%token END

/**
 * Tokens para números, ids e strings
 */
%token <sval> ID		"identifier"
%token <sval> STRING	"string"
%token <ival> INTEGER	"integer number"
%token <fval> REAL		"real number"

/**
 * Tokens para símbolos reservados
 */
%token COMMA		","
%token RIGHT_BRACES	"{"
%token LEFT_BRACES	"}"
%token QUESTION		"?"

/**
 * Tokens para palavras reservadas
 */
%token RELATION		"word relation"
%token ATTRIBUTE	"word attribute"
%token DATA			"word data"
%token TYPE_NUMERIC	"word numeric"
%token TYPE_STRING	"word string"
%token TYPE_DATE	"word date"

/**
 * Tipos para os não-terminais
 */
%type <nominal> nominal
%type <nominal> more_nominal
%type <value>   value
%type <row>	 more_values
%type <row>	 more_sparse

%destructor { delete $$; } <sval> <nominal> <row> <value>

%%

/**
 * Regras sintáticas
 */
main
	: header {saveLine;} data
	;

header
	: {saveLine;} relation attributes
	| error { throwError(SIN_INVALID_RELATION, driver); }
	;

relation
	: RELATION ID {
		driver.dataset->setName($2);
		delete $2;
	}
	;

attributes
	: {saveLine;} attribute attributes
	| error { throwError(SIN_INVALID_ATTRIBUTE, driver); }
	|
	;

attribute
	: ATTRIBUTE ID TYPE_NUMERIC {
		driver.dataset->addAttribute(new Attribute(*$2, NUMERIC));
		delete $2;
	}
	| ATTRIBUTE ID TYPE_STRING {
		driver.dataset->addAttribute(new Attribute(*$2, STRING));
		delete $2;
	}
	| ATTRIBUTE ID nominal {
		driver.dataset->addAttribute(new Attribute(*$2, NOMINAL, *$3));
		delete $2;
		delete $3;
	}
	| ATTRIBUTE ID TYPE_DATE STRING {
		driver.dataset->addAttribute(new Attribute(*$2, DATE, *$4));
		delete $2;
		delete $4;
	}
	;

nominal
	: RIGHT_BRACES ID more_nominal LEFT_BRACES {
		$$ = $3;
		$$->push_front(*$2);
		delete $2;
	}
	;

more_nominal
	: COMMA ID more_nominal {
		$$ = $3;
		$$->push_front(*$2);
		delete $2;
	}
	| { $$ = new Nominal; }
	;

data
	: DATA instances
	;

instances
	: {saveLine;} instance instances
	| error { throwError(SIN_INVALID_INSTANCE, driver); }
	|
	;

instance
	: value more_values {
		DataList* row = $2;
		row->push_front($1);
		driver.dataset->addInstance(row, false);
		delete row;
	}
	| RIGHT_BRACES INTEGER value more_sparse LEFT_BRACES {
		DataList* row = $4;
		$3->setIndex($2);
		row->push_front($3);
		driver.dataset->addInstance(row, true);
		delete row;
	}
	;

more_values
	: COMMA value more_values {
		$$ = $3;
		$$->push_front($2);
	}
	| { $$ = new DataList; }
	;

more_sparse
	: COMMA INTEGER value more_sparse {
		$$ = $4;
		$3->setIndex($2);
		$$->push_front($3);
	}
	| { $$ = new DataList; }
	;

value
	: INTEGER {
		$$ = new Value(NUMERIC, $1);
	}
	| REAL {
		$$ = new Value(NUMERIC, $1);
	}
	| STRING {
		$$ = new Value(STRING, *$1);
		delete $1;
	}
	| ID {
		$$ = new Value(NOMINAL, *$1);
		delete $1;
	}
	| QUESTION {
		$$ = new Value(EMPTY);
	}
	;

%%

#include "arff/Driver.h"
#include "arff/Scanner.h"

//===========================================================================//

int yylex(Parser::semantic_type* yylval, Parser::location_type* yyloc,
		Driver &driver)
{
	return driver.scanner->yylex(yylval, yyloc);
}

//===========================================================================//

void markLine(Driver &driver)
{
	driver.scanner->markLine();
}

//===========================================================================//

void throwError(ErrorType error, Driver &driver)
{
	throw ParallelMLPException(error, driver.scanner->getToken(),
			driver.scanner->getLineno());
}

//===========================================================================//

void Parser::error(const Parser::location_type &loc, const string &msg)
{

}

//===========================================================================//

