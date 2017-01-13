#include <signal.h>
#include "Common.h"
#include "Log.h"
#include "Looper.h"
#include "RedisIO.h"
#include "VectorNetSubscriber.h"
#include "VectorSubscriber.h"
#include "VectorClassFinder.h"

#ifdef __cplusplus
extern "C" {
#endif

char *label_array[11] = {"SeonghoBaek", "ByongrakSeo", "HyungkiNoh", "kiyoungKim", "MinsamKo", "YonbeKim", "DaeyoungPark", "JangHyungLee", "KwangheeLee", "SanghoonLee", "Unknown"};

extern int G_LABEL_INDEX;

VectorClassFinder *pGVCF = NULL;

int run_identifier() {
    //signal(SIGPIPE, SIG_IGN);

    //Looper *pLooper = new Looper(callback);

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

    pGVCF = new VectorClassFinder();

    pGVCF->startThread();

    sleep(1);
#if 0
    // Using REDIS
    //VectorSubscriber *vs = new VectorSubscriber("127.0.0.1", 6379, "vector");
#else
    // TCP Socket
    VectorNetSubscriber *vs = new VectorNetSubscriber("127.0.0.1", 55555, pGVCF);
#endif

    vs->startThread();

    sleep(1);
    //pLooper->wait(-1);

    return 0;
}

char *get_lable_in_box(int left, int top, int right, int bottom)
{
    int center_x = -1;
    int center_y = -1;

    //center_y = (bottom + top) / 2;
    //center_x = (right + left) / 2;

    //return pGVCF->getClosestLabel(center_x, center_y);
    return pGVCF->getClosestIoULabel(left, right, top, bottom);
}

void version_up()
{
    pGVCF->versionUp();
}

#ifdef __cplusplus
}
#endif
