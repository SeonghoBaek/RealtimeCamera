//
// Created by major on 17. 1. 6.
//

#include "VectorClassFinder.h"
#include "Log.h"

int VectorClassFinder::nodtify(float data1, Vector& vector) {
    int numQueue;

    LOCK(this->mFrameLock)
    {
        this->mNextFrame = vector.mFrame;

        LOGD("Next Frame: %d, QSize: %d", this->mNextFrame, this->mVQ.getSize());

        if (this->mCurrentFrame != -1)
        {
            if (this->mNextFrame != this->mCurrentFrame)
            {
                this->mVQ.clear();
                this->mCurrentFrame = -1;
            }
        }
    }

    this->mVQ.push(vector);

    this->mpLooper->wakeup();

    return 0;
}

void VectorClassFinder::run()
{
    this->mpLooper->wait(-1);
}

int VectorClassFinder::looperCallback(const char *event) {
    // No need to check event.
    Vector *pV = NULL;

    //LOGD("Queue: %d, Frame: %d", this->mVQ.getSize(), pV->mFrame);

    pV = this->mVQ.pop();

    if (pV->mpData)
    {
        LOCK(this->mFrameLock)
        {
            this->mCurrentFrame = pV->mFrame;
        }

        LOGI("Current Frame: %d", this->mCurrentFrame);

        // Process.
        // For queue latency test
        //sleep(3);

    }

    /*
    redisReply *pReply = (redisReply *)redisCommand(this->mpRedisIO->getContext(), "SMEMBERS user");

    if (pReply->type == REDIS_REPLY_ARRAY)
    {
        for (int i = 0; i < pReply->elements; i++)
        {
            //LOGI("%u) %s", i, pR->element[i]->str);

            if (pReply->element[i]->str)
            {
                LOGI("REDIS: %s", pReply->element[i]->str);
            }
        }
    }

    freeReplyObject(pReply);
    */

    delete pV;

    return 0;
}
