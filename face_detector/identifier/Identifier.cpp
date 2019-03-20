#include <signal.h>
#include "Common.h"
#include "Log.h"
#include "Looper.h"
#include "RedisIO.h"
#include "VectorNetSubscriber.h"
#include "VectorSubscriber.h"
#include "VectorClassFinder.h"

#ifdef USE_SRC
#include "../sparse/sparse.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

VectorClassFinder *pGVCF = NULL;

int run_identifier(const char* local_server) {
    signal(SIGPIPE, SIG_IGN);

    pGVCF = new VectorClassFinder();

    pGVCF->startThread();

    sleep(1);

    // TCP Socket
    VectorNetSubscriber *vs = new VectorNetSubscriber(local_server, 55555, pGVCF);
    VectorNetSubscriber *vs2 = new VectorNetSubscriber(local_server, 55556, pGVCF);

    vs->startThread();
    vs2->startThread();

    sleep(1);

#ifdef USE_SRC
    init_src_model();
#endif

    return 0;
}

void clear_label_check_info()
{
    pGVCF->resetLabelCheck();
}

char *get_label_in_box(int left, int top, int right, int bottom)
{
    return pGVCF->getClosestIoULabel(left, right, top, bottom);
}

void version_up()
{
    pGVCF->versionUp();
}

void invalidate()
{
    pGVCF->invalidate();
}

int door_open() {
    return pGVCF->fireUserEvent(-1);
}

#ifdef USE_SRC
void train_sparse()
{
    train(0, NULL);
}

void test_sparse()
{
    test(0, NULL);
}

int test_image_file(const char *image_file_path, const char *label)
{
    return src_test_file(image_file_path, label);
}
#endif

#ifdef __cplusplus
}
#endif
