#include "arff/Driver.h"
#include "arff/Scanner.h"
#include "arff/Parser.hh"
#include "arff/Relation.h"

namespace ParallelMLP
{

//===========================================================================//

Driver::Driver(const string &filename)
{
	this->filename = filename;

	istream.open(filename.c_str());
	if (!istream.good())
	{
		cerr << "Couldn't open file " << filename << endl;
		exit(EXIT_FAILURE);
	}

	dataset = new Relation(*this);
	scanner = new Scanner(*this);
	parser = new Parser(*this);
}

//===========================================================================//

Driver::~Driver()
{
	delete dataset;
	delete scanner;
	delete parser;
}

//===========================================================================//

Relation* Driver::parse()
{
	parser->parse();
	return dataset;
}

//===========================================================================//

}
