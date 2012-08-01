#include "mlp/Settings.h"

namespace MLP
{

//===========================================================================//

Settings::Settings(uint nLayers)
{
	this->nLayers = nLayers;
	units = new uint[nLayers + 1];
}

//===========================================================================//

Settings::~Settings()
{
	delete units;
}

//===========================================================================//

}
