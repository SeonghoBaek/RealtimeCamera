//
// Created by major on 17. 1. 6.
//

#include "VectorClassFinder.h"
#include "Log.h"
#include <math.h>

#define RESET_FREQ 60
#define USE_UPDATE_CHECK 0

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
    if (b <= t) return 0.0;

    int interArea = (r - l + 1) * (b - t + 1);
    int boxAArea = (right - left + 1) * (bottom - top + 1);
    int boxBArea = (sright - sleft + 1) * (sbottom - stop + 1);
    float iou = (float)interArea / (float)(boxAArea + boxBArea - interArea);

	/*
    if (iou >= 1)
    {
        LOGD("sl: %d, sr: %d, st: %d, sb: %d", sleft, sright, stop, sbottom);
        LOGD("tl: %d, tr: %d, tt: %d, tb: %d", left, right, top, bottom);
        LOGD("l: %d, r: %d, t: %d, b: %d", l, r, t, b);
        LOGD("boxA: %d, boxB: %d, Inter: %d", boxAArea, boxBArea, interArea);
    }
	*/

    return iou;
}

void VectorClassFinder::resetLabelCheck()
{
    LOCK(this->mFrameLock)
    {
        for (int i = 0; i < this->mNumLabel; i++)
        {
            this->mpLabels[i].mChecked = 0;
        }
    }
}

char* VectorClassFinder::getClosestIoULabel(int left, int right, int top, int bottom)
{
    float IoU = 0.4;
    int   index = -1;

    //gResetTime++;
    //gResetTime %= RESET_FREQ;

    //LOGD("Check IOU");

#if (USE_UPDATE_CHECK == 1)
    struct timeval time;

    double  cur;

    if (gettimeofday(&time,NULL))
    {
        cur =  0;
    }
    else
    {
        cur = (double) time.tv_sec;
    }
#endif

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
                //LOGD("X: %d, checked: %d", this->mpLabels[i].getX(), this->mpLabels[i].mChecked);
                if (this->mpLabels[i].getX() > 0 && this->mpLabels[i].mChecked == 0)
                {

                    if (abs(this->mpLabels[i].mVersion - this->mVersion) > 15)
                    {

                        this->mpLabels[i].setX(-1);
                        LOGD("Version diff: Disappeared? %s", this->mpLabels[i].getLabel());
                        continue;
                    }

                    float iou = this->getIoU(left, right, top, bottom,
                                             this->mpLabels[i].mLeft,  this->mpLabels[i].mRight,  this->mpLabels[i].mTop,  this->mpLabels[i].mBottom);

                    //LOGD("iou: %f", iou);

                    if (iou >= IoU)
                    {
#if (USE_UPDATE_CHECK == 1)
                        if (cur - this->mpLabels[i].mUpdateTime > 4.0)
                        {
                            LOGD("diff: %f for %s", (float)(cur - this->mpLabels[i].mUpdateTime), this->mpLabels[i].getLabel());
                            this->mpLabels[i].setX(-1);
                            continue;
                        }
#endif
                        index = i;
                        IoU = iou;
                        break;
                    }
                }
            }

            if (index != -1)
            {
                this->mpLabels[index].mRight = right;
                this->mpLabels[index].mLeft = left;
                this->mpLabels[index].mTop = top;
                this->mpLabels[index].mBottom = bottom;

                this->mpLabels[index].mVersion = this->mVersion;
                this->mpLabels[index].mChecked = 1;

                //LOGD("OK Still %s", this->mpLabels[index].getLabel());
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

#if (USE_UPDATE_CHECK == 1)
    struct timeval time;
    double  cur;

    if (gettimeofday(&time,NULL))
    {
        cur =  0;
    }
    else
    {
        cur = (double) time.tv_sec;
    }
#endif

    LOCK(this->mFrameLock)
    {
        if (this->mpLabels[pV->mLabelIndex].getX() != -1)
        {
#if 1
            this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
#if (USE_UPDATE_CHECK == 1)
            this->mpLabels[pV->mLabelIndex].mUpdateTime = cur;
#endif
#else
            float iou = this->getIoU(pV->mX, pV->mY, pV->mT, pV->mB,
                                     this->mpLabels[pV->mLabelIndex].mLeft, this->mpLabels[pV->mLabelIndex].mRight,
                                     this->mpLabels[pV->mLabelIndex].mTop, this->mpLabels[pV->mLabelIndex].mBottom);

            if (iou > 0.3)
            {
                //LOGD("Still same person: %d, %f", pV->mLabelIndex, iou);
                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
#if (USE_UPDATE_CHECK == 1)
                this->mpLabels[pV->mLabelIndex].mUpdateTime = cur;
#endif
            }
            else
            {
                LOGD("Reset Index: %s, %f", this->mpLabels[pV->mLabelIndex].getLabel(), iou);
                this->mpLabels[pV->mLabelIndex].setX(-1); // Give a more try
            }
#endif
        }
        else
        {
            bool tryNext = FALSE;

            LOGD("New : %s, Confidence: %f", this->mpLabels[pV->mLabelIndex].getLabel(), pV->mConfidence);

            //if (pV->mConfidence > 0.9)
            {
                for (int i = 0; i < this->mNumLabel; i++)
                {
                    if (this->mpLabels[i].getX() > 0)
                    {
                        float iou = this->getIoU(pV->mX, pV->mY, pV->mT, pV->mB,
                                                 this->mpLabels[i].mLeft,
                                                 this->mpLabels[i].mRight,
                                                 this->mpLabels[i].mTop,
                                                 this->mpLabels[i].mBottom);

                        LOGD("IoU: %f, New : %s, Cur: %s", iou, this->mpLabels[pV->mLabelIndex].getLabel(), this->mpLabels[i].getLabel());

                        if (iou > 0.5)
                        {
                            LOGD("Confidence: %f, %f", pV->mConfidence, this->mpLabels[i].mConfidence);
                            if (this->mpLabels[i].mConfidence < pV->mConfidence)
                            {
                                LOGD("Next try: %s", this->mpLabels[i].getLabel());
                                this->mpLabels[i].setX(-1); // Give a more try;
                            }

                            tryNext = TRUE;
                        }
                    }
                }

                if (tryNext == FALSE) {
                    LOGD("User %s added", this->mpLabels[pV->mLabelIndex].getLabel());
                    this->mpLabels[pV->mLabelIndex].setX(1);
                    this->mpLabels[pV->mLabelIndex].setY(1);
                    this->mpLabels[pV->mLabelIndex].mLeft = pV->mX;
                    this->mpLabels[pV->mLabelIndex].mRight = pV->mY;
                    this->mpLabels[pV->mLabelIndex].mTop = pV->mT;
                    this->mpLabels[pV->mLabelIndex].mBottom = pV->mB;
                    this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
                    this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;
                    this->mpLabels[pV->mLabelIndex].mChecked = 0;
#if (USE_UPDATE_CHECK == 1)
                    this->mpLabels[pV->mLabelIndex].mUpdateTime = cur;
#endif
                }
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
