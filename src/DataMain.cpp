#include "arff/Driver.h"
#include "arff/Relation.h"
#include "database/RelationAdapter.h"
#include "database/ExampleSetAdapter.h"

using namespace ARFF;
using namespace Database;

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        cerr << "Usage mode: " << argv[0] << " <input file>" << endl;
        return EXIT_FAILURE;
    }

    string input(argv[1]);
    Driver driver(input);

    try
    {
        Relation* relation = driver.parse();
        RelationAdapter::insert(*relation);
    }
    catch(exception &ex)
    {
        cerr << ex.what() << endl;
    }

    return EXIT_SUCCESS;
}
