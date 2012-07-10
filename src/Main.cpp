#include "arff/Driver.h"
#include "arff/DataSet.h"
#include <cstdlib>

using namespace ARFF;

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
        DataSet* dataset = driver.parse();

        cout << "Relation: " << dataset->relation << endl;

        for(AttributePtr attr : dataset->attributes)
        {
            cout << "Attribute " << attr->name << ": type/" << attr->type;
            if(attr->type == NOMINAL)
            {
                cout << " values: ";
                for(string &str : *(attr->nominal))
                    cout << str << " ";
            }
            cout << endl;
        }

        cout << "Data" << endl;
        for(InstancePtr row : dataset->data)
        {
            for(ValuePtr value : *row)
            {
                if(value->type == NUMERIC)
                    cout << value->number;
                else if(value->type == NOMINAL)
                    cout << *(value->str);
                cout << "/" << value->type << " ";
            }
            cout << endl;
        }
    }
    catch(exception &ex)
    {
        cerr << ex.what() << endl;
    }

    return EXIT_SUCCESS;
}
