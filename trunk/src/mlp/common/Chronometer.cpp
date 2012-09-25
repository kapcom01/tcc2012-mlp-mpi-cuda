#include "mlp/common/Chronometer.h"

namespace ParallelMLP
{

//===========================================================================//

Chronometer::Chronometer()
{
    reset();
}

//===========================================================================//

void Chronometer::reset()
{
    clock_gettime(CLOCK_REALTIME, &init);
}

//===========================================================================//

double Chronometer::getSeconds()
{
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    double time = (end.tv_sec - init.tv_sec) + (end.tv_nsec - init.tv_nsec)
    		/ (double) 1000000000;
    return time;
}

//===========================================================================//

double Chronometer::getMiliseconds()
{
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    double time = (end.tv_sec - init.tv_sec) * 1000
    		+ (end.tv_nsec - init.tv_nsec) / (double) 1000000;
    return time;
}

}
