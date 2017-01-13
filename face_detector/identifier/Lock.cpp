#include "Common.h"
#include "Lock.h"

Mutex_t Lock::createMutex()
{
    pthread_mutex_t *m = new pthread_mutex_t;

    pthread_mutex_init(m, NULL);

    return (Mutex_t)m;
}

Mutex_t Lock::createMutex(bool startLock)
{
    pthread_mutex_t *m = new pthread_mutex_t;

    pthread_mutex_init(m, NULL);

    if (startLock == TRUE)
    {
        pthread_mutex_lock(m);
    }

    return (Mutex_t)m;
}

int Lock::unlockMutex(Mutex_t mutex)
{
    return pthread_mutex_unlock(mutex);
}

int Lock::deleteMutex(Mutex_t mutex)
{
    pthread_mutex_unlock(mutex);
    pthread_mutex_destroy(mutex);

    delete mutex;

    return 0;
}
