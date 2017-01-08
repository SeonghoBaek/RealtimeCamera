#include <signal.h>
#include "Common.h"
#include "Log.h"
#include "Looper.h"
#include "RedisIO.h"
#include "VectorNetSubscriber.h"
#include "VectorSubscriber.h"
#include "VectorClassFinder.h"

int callback(const char* msg)
{
    LOGI(msg);

    return 0;
}

int main(int argc, char **argv)
{
    signal(SIGPIPE, SIG_IGN);

    Looper *pLooper = new Looper(callback);

    /*
    if (argc < 4)
    {
        LOGI("usage: %s <server> <port> <channel>", argv[0]);

        return -1;
    }

    const char* pServer = argv[1];
    int port = atoi(argv[2]);
    const char* pChannel = argv[3];
    */

    VectorClassFinder *pVCF = new VectorClassFinder();

    pVCF->startThread();


#if 0
    // Using REDIS
    //VectorSubscriber *vs = new VectorSubscriber("127.0.0.1", 6379, "vector");
#else
    // TCP Socket
    VectorNetSubscriber *vs = new VectorNetSubscriber("127.0.0.1", 55555, pVCF);
#endif

    vs->startThread();

    pLooper->wait(-1);

    return 0;
}

