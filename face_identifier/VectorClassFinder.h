//
// Created by major on 17. 1. 6.
//

#ifndef REALTIMECAMERA_VECTORCLASSFINDER_H
#define REALTIMECAMERA_VECTORCLASSFINDER_H

#include "Common.h"
#include "VectorQueue.h"
#include "Thread.h"
#include "Looper.h"
#include "RedisIO.h"
#include "Lock.h"

class VectorClassFinder:public IVectorNotifier, public Thread, public ILooper
{
private:
    VectorQueue mVQ;
    Looper      *mpLooper;
    RedisIO     *mpRedisIO;
    int         mCurrentFrame;
    int         mNextFrame;
    Mutex_t     mFrameLock;

public:
    IMPLEMENT_THREAD(run());

    VectorClassFinder()
    {
        mpLooper = new Looper(this);
        mpRedisIO = new RedisIO();
        mpRedisIO->connect(2);
        mCurrentFrame = -1;
        mNextFrame = -1;
        mFrameLock = Lock::createMutex();
    }

    virtual ~VectorClassFinder()
    {
        if (mFrameLock)
        {
            Lock::deleteMutex(this->mFrameLock);
        }

        if (mpLooper)
        {
            delete mpLooper;
        }

        if (mpRedisIO)
        {
            delete mpRedisIO;
        }
    }

    int nodtify(float data1, Vector& vector) override;

    int looperCallback(const char *event) override;

    void run();
};

#endif //REALTIMECAMERA_VECTORCLASSFINDER_H
