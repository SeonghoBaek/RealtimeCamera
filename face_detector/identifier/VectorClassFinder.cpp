//
// Created by major on 17. 1. 6.
//

#include "VectorClassFinder.h"
#include "Log.h"
#include <math.h>

#define RESET_FREQ 60

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

int gResetTime = 0;

float VectorClassFinder::getDistance(int sx, int sy, int tx, int ty)
{
    float distance = 0.0;

    distance = sqrtf((sx - tx)*(sx - tx) + (sy - ty)*(sy - ty));

    return distance;
}

float VectorClassFinder::getIoU(int sleft, int sright, int stop, int sbottom, int left, int right, int top, int bottom)
{
    int l = MAX(left, sleft);
    int r = MIN(right, sright);
    int t = MAX(top, stop);
    int b = MIN(bottom, sbottom);

    if (r <= l) return 0.0;

    int interArea = (r - l + 1) * (b - t + 1);
    int boxAArea = (right - left + 1) * (bottom - top + 1);
    int boxBArea = (sright - sleft + 1) * (sbottom - stop + 1);
    float iou = (float)interArea / (float)(boxAArea + boxBArea - interArea);

    if (iou >= 1)
    {
        LOGD("sl: %d, sr: %d, st: %d, sb: %d", sleft, sright, stop, sbottom);
        LOGD("tl: %d, tr: %d, tt: %d, tb: %d", left, right, top, bottom);
        LOGD("l: %d, r: %d, t: %d, b: %d", l, r, t, b);
        LOGD("boxA: %d, boxB: %d, Inter: %d", boxAArea, boxBArea, interArea);
    }

    return iou;
}

char* VectorClassFinder::getClosestIoULabel(int left, int right, int top, int bottom)
{
    float IoU = 0.5;
    int   index = -1;

    //gResetTime++;
    //gResetTime %= RESET_FREQ;

    //LOGD("Check IOU");

    /*
    if (gResetTime == 0)
    {
        LOCK(this->mFrameLock)
        {
            for (int i = 0; i < this->mNumLabel; i++)
            {
                this->mpLabels[i].setX(-1);
            }
        }
    }
    else
    */
    {
        LOCK(this->mFrameLock)
        {
            for (int i = 0; i < this->mNumLabel; i++)
            {
                if (this->mpLabels[i].getX() > 0)
                {
                    if (abs(this->mpLabels[i].mVersion - this->mVersion) > 1)
                    {

                        this->mpLabels[i].setX(-1);
                        continue;
                    }

                    float iou = this->getIoU(left, right, top, bottom,
                                             this->mpLabels[i].mLeft,  this->mpLabels[i].mRight,  this->mpLabels[i].mTop,  this->mpLabels[i].mBottom);

                    //LOGD("iou: %f", iou);

                    if (iou > IoU)
                    {
                        index = i;
                        IoU = iou;
                    }
                }
            }

            //LOGD("for exit");

            if (index != -1)
            {
                this->mpLabels[index].mRight = right;
                this->mpLabels[index].mLeft = left;
                this->mpLabels[index].mTop = top;
                this->mpLabels[index].mBottom = bottom;

                this->mpLabels[index].mVersion = this->mVersion;
            }
        }
    }

    //LOGD("index = %d", index);

    if (index == -1) return "";

    return this->mpLabels[index].getLabel();
}

char* VectorClassFinder::getClosestLabel(int center_x, int center_y)
{
    float closest = 0x0FFFFFFF;
    int   index = -1;

    gResetTime++;
    gResetTime %= RESET_FREQ;


    /*
    if (gResetTime == 0)
    {
        LOCK(this->mFrameLock)
        {
            for (int i = 0; i < this->mNumLabel; i++)
            {
                this->mpLabels[i].setX(-1);
            }
        }
    }
    else
    */
    {
        LOCK(this->mFrameLock)
        {
            for (int i = 0; i < this->mNumLabel; i++)
            {
                if (this->mpLabels[i].getX() > 0) {
                    float dist = getDistance(this->mpLabels[i].getX(), this->mpLabels[i].getY(), center_x, center_y);

                    LOGD("%d,%d to %d,%d, INDEX: %d, DIST: %f", this->mpLabels[i].getX(), this->mpLabels[i].getY(),
                         center_x, center_y, i, dist);

                    if (dist < closest) {
                        closest = dist;
                        index = i;
                    }
                }
            }

            if (closest > 78) {
                index = -1;
                this->mpLabels[index].setX(-1);
            }

            if (index != -1) {
                this->mpLabels[index].setX(center_x);
                this->mpLabels[index].setY(center_y);
                this->mpLabels[index].versionUp();
            }
        }
    }

    if (index == -1) return "";

    return this->mpLabels[index].getLabel();
}

int VectorClassFinder::nodtify(float data1, Vector& vector) {
    int numQueue;

    /*
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
    */

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

    //LOGD("ID: %d, X: %d, Y: %d, C: %f", pV->mLabelIndex, pV->mX, pV->mY, pV->mConfidence);

    LOCK(this->mFrameLock)
    {
        if (this->mpLabels[pV->mLabelIndex].getX() != -1)
        {
            float iou = this->getIoU(pV->mX, pV->mY, pV->mT, pV->mB,
                                     this->mpLabels[pV->mLabelIndex].mLeft, this->mpLabels[pV->mLabelIndex].mRight,
                                     this->mpLabels[pV->mLabelIndex].mTop, this->mpLabels[pV->mLabelIndex].mBottom);

            //LOGD("updated iou: %f",iou);

            if (iou > 0.5)
            {
                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
            }
            else
            {
                this->mpLabels[pV->mLabelIndex].setX(-1); // Give a more try
            }
        }
        else
        {
            bool tryNext = FALSE;

            for (int i = 0; i < this->mNumLabel; i++)
            {
                if (this->mpLabels[i].getX() > 0)
                {
                    float iou = this->getIoU(pV->mX, pV->mY, pV->mT, pV->mB,
                                             this->mpLabels[pV->mLabelIndex].mLeft, this->mpLabels[pV->mLabelIndex].mRight,
                                             this->mpLabels[pV->mLabelIndex].mTop, this->mpLabels[pV->mLabelIndex].mBottom);

                    if (iou > 0.5)
                    {
                        if (this->mpLabels[i].mConfidence < pV->mConfidence)
                        {
                            this->mpLabels[i].setX(-1); // Give a more try;
                        }

                        tryNext = TRUE;
                    }
                }
            }

            if (tryNext == FALSE)
            {
                this->mpLabels[pV->mLabelIndex].setX(pV->mX);
                this->mpLabels[pV->mLabelIndex].setY(pV->mY);
                this->mpLabels[pV->mLabelIndex].mLeft = pV->mX;
                this->mpLabels[pV->mLabelIndex].mRight = pV->mY;
                this->mpLabels[pV->mLabelIndex].mTop = pV->mT;
                this->mpLabels[pV->mLabelIndex].mBottom = pV->mB;
                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
                this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;
            }
        }
    }

    /*
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
    */

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
