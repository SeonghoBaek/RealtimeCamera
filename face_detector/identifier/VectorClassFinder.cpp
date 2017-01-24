//
// Created by major on 17. 1. 6.
//

#include "VectorClassFinder.h"
#include "Log.h"
#include <math.h>
#include <hiredis.h>
#include "../sparse/sparse.h"

#define RESET_FREQ 60
#define USE_UPDATE_CHECK 0
#define USE_LINED_LIST 1
#define IOU_SINGLE_MODE 0

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#if (IOU_SINGLE_MODE == 1)
float IOU_INTERSECT_NEW_USER = 0.7;
float IOU_INTERSECT_CUR_USER = 0.4;
float IOU_INTERSECT_TRACKING = 0.4;
#else
float IOU_INTERSECT_NEW_USER = 0.5;
float IOU_INTERSECT_CUR_USER = 0.4;
float IOU_INTERSECT_TRACKING = 0.4;
#endif

redisContext    *gRedisContext = NULL;

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

#if (IOU_SINGLE_MODE == 1)
    int interArea = (r - l + 1) * (b - t + 1);
    int boxAArea = (sright - sleft + 1) * (sbottom - stop + 1);
    float iou = (float)interArea / (float)(boxAArea);
#else
    int interArea = (r - l + 1) * (b - t + 1);
    int boxAArea = (right - left + 1) * (bottom - top + 1);
    int boxBArea = (sright - sleft + 1) * (sbottom - stop + 1);
    float iou = (float)interArea / (float)(boxAArea + boxBArea - interArea);
#endif

    return iou;
}

void VectorClassFinder::resetLabelCheck()
{
#if (USE_LINED_LIST == 1)
    LOCK(this->mFrameLock)
    {
        LabelListItem *pItem = this->mpActiveLabelList;

        while (pItem != NULL)
        {
            pItem->mpLabel->mChecked = 0;
            pItem = pItem->mpNext;
        }
    }
#else
    LOCK(this->mFrameLock)
    {
        for (int i = 0; i < this->mNumLabel; i++)
        {
            this->mpLabels[i].mChecked = 0;
        }
    }
#endif
}

void VectorClassFinder::updateLabel(Label *pLabel, Vector *pV)
{
    pLabel->mConfidence = pV->mConfidence;
#if 1
    pLabel->mLeft = pV->mX;
    pLabel->mRight = pV->mY;
    pLabel->mTop = pV->mT;
    pLabel->mBottom = pV->mB;
#endif
}

char* VectorClassFinder::getClosestIoULabel(int left, int right, int top, int bottom)
{
    int   max_iou_index = -1;
    float IOU = IOU_INTERSECT_TRACKING;

#if (USE_UPDATE_CHECK == 1)
    struct timeval time;

    double  cur;

    if (gettimeofday(&time,NULL))
    {
        cur =  0;
    }
    else
    {
        cur = (double)time.tv_sec + (double)time.tv_usec * .000001;
    }
#endif

#if (USE_LINED_LIST == 1)
    LOCK(this->mFrameLock)
    {
        LabelListItem *pItem = this->mpActiveLabelList;

        while (pItem != NULL)
        {
            if (pItem->mpLabel->mChecked == 0)
            {
                //LOGD("Version: %d", this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion);

                if (abs(this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion - this->mVersion) > RESET_FREQ)
                {
                    LOGD("Version diff: Too old or Disappeared? %s", this->mpLabels[pItem->mpLabel->mLabelIndex].getLabel());

                    this->mpLabels[pItem->mpLabel->mLabelIndex].setX(-1);

                    LabelListItem *pDelItem = pItem;

                    if (pItem == this->mpActiveLabelList) // Head
                    {
                        LOGD("Delete Head");
                        this->mpActiveLabelList = pItem->mpNext;

                        if (this->mpActiveLabelList != NULL)
                        {
                            this->mpActiveLabelList->mpPrev = this->mpActiveLabelList;
                        }

                        delete pDelItem;

                        pItem = this->mpActiveLabelList;

                        continue;
                    }
                    else
                    {
                        pItem->mpPrev->mpNext = pItem->mpNext;

                        if (pItem->mpNext != NULL)
                        {
                            pItem->mpNext->mpPrev = pItem->mpPrev;
                        }

                        pItem = pItem->mpNext;

                        delete pDelItem;

                        continue;
                    }
                }
                else
                {
                    int labelIndex = pItem->mpLabel->mLabelIndex;

                    float iou = this->getIoU(left, right, top, bottom,
                                             this->mpLabels[labelIndex].mLeft,  this->mpLabels[labelIndex].mRight,
                                             this->mpLabels[labelIndex].mTop, this->mpLabels[labelIndex].mBottom);

                    //LOGD("iou: %f", iou);

                    if (iou >= IOU)
                    {
                        max_iou_index = labelIndex;
                        IOU = iou;

                        //LOGD("OK Still %s", this->mpLabels[index].getLabel());
                    }
                }
            }

            if (pItem != NULL) pItem = pItem->mpNext;
        }
    }
#else

    LOCK(this->mFrameLock)
    {
        for (int i = 0; i < this->mNumLabel; i++)
        {
            //LOGD("X: %d, checked: %d", this->mpLabels[i].getX(), this->mpLabels[i].mChecked);
            if (this->mpLabels[i].getX() >= 0 && this->mpLabels[i].mChecked == 0)
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


#endif
    //LOGD("index = %d", index);

    if (max_iou_index == -1)
    {
        return "";
    }
    else
    {
        this->mpLabels[max_iou_index].mRight = right;
        this->mpLabels[max_iou_index].mLeft = left;
        this->mpLabels[max_iou_index].mTop = top;
        this->mpLabels[max_iou_index].mBottom = bottom;
        this->mpLabels[max_iou_index].mVersion = this->mVersion;
        this->mpLabels[max_iou_index].mChecked = 1;

#if (USE_UPDATE_CHECK == 1)
        LOGD("latency: %f for %s", (float)(cur - this->mpLabels[max_iou_index].mUpdateTime), this->mpLabels[max_iou_index].getLabel());
#endif
    }

    return this->mpLabels[max_iou_index].getLabel();
}

int VectorClassFinder::nodtify(float data1, Vector& vector)
{
    this->mVQ.push(vector);

    this->mpLooper->wakeup();

    return 0;
}

void VectorClassFinder::run()
{

    const char *hostname = "127.0.0.1";
    int port = 6379;
    struct timeval timeout = {1, 500000};

    gRedisContext = redisConnectWithTimeout(hostname, port, timeout);

    if (gRedisContext == NULL || gRedisContext->err)
    {
        if (gRedisContext)
        {
            LOGE("Connection Error: %s\n", gRedisContext->errstr);
            redisFree(gRedisContext);
            gRedisContext = NULL;
        } else {
            LOGE("Connection error: can't allocate redis context\n");
        }
    }

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
        cur = (double) time.tv_sec + (double)time.tv_usec * .000001;
    }
#endif

#if (USE_LINED_LIST == 1)

    LOCK(this->mFrameLock)
    {
        if (this->mpActiveLabelList == NULL)
        {
            if (pV->mConfidence < 0.9)
            {
                delete pV;

                return 0;
            }

            LOGD("Add New Face: %s",  this->mpLabels[pV->mLabelIndex].getLabel());

            LabelListItem *pItem = new LabelListItem();

            if (pItem == NULL)
            {
                delete pV;

                return -1;
            }

            pItem->mpLabel = new Label();

            if (pItem->mpLabel == NULL)
            {
                delete pItem;

                return -1;
            }

            Label *pNewLabel = pItem->mpLabel;

            pNewLabel->mLabelIndex = pV->mLabelIndex;
            this->mpLabels[pV->mLabelIndex].setX(1);
            this->mpLabels[pV->mLabelIndex].setY(1);
            this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;

            this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);

            this->mpActiveLabelList = pItem;

            if (gRedisContext)
            {
                char id[3];
                sprintf(id, "%d", pV->mLabelIndex);

                redisCommand(gRedisContext, "PUBLISH %s %s", "tts", id);
            }

#if (USE_UPDATE_CHECK == 1)
            pNewLabel->mUpdateTime = cur;
            this->mpLabels[pV->mLabelIndex].mUpdateTime = cur; // For fast check.
#endif
        }
        else
        {
            if (this->mpLabels[pV->mLabelIndex].getX() == 1)
            {
                this->mpLabels[pV->mLabelIndex].setX(1); // For explicit action
                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;

                //this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);

                this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;
                //LOGD("Update %s confidence %f", this->mpLabels[pV->mLabelIndex].getLabel(), pV->mConfidence);

#if (USE_UPDATE_CHECK == 1)
                this->mpLabels[pV->mLabelIndex].mUpdateTime = cur;
#endif
            }
            else if (this->mpLabels[pV->mLabelIndex].getX() == 0)
            {
                this->mpLabels[pV->mLabelIndex].setX(1);
                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
                //this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);
                this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;
                LOGD("Still: %s", this->mpLabels[pV->mLabelIndex].getLabel());
            }
            else // -1
            {
                LabelListItem *pHead = this->mpActiveLabelList;

                while (pHead != NULL)
                {
                    int labelIndex = pHead->mpLabel->mLabelIndex;

                    float iou = this->getIoU(this->mpLabels[labelIndex].mLeft,
                                             this->mpLabels[labelIndex].mRight,
                                             this->mpLabels[labelIndex].mTop,
                                             this->mpLabels[labelIndex].mBottom,
                                             pV->mX, pV->mY, pV->mT, pV->mB);

                    LOGD("IoU: %f, New : %s, Cur: %s", iou, this->mpLabels[pV->mLabelIndex].getLabel(), this->mpLabels[labelIndex].getLabel());

                    if (iou > IOU_INTERSECT_NEW_USER)
                    {
                        LOGD("Confidence New: %f, Cur: %f", pV->mConfidence, this->mpLabels[labelIndex].mConfidence);
                        if (this->mpLabels[labelIndex].mConfidence < pV->mConfidence)
                        {
                            if (this->mpLabels[labelIndex].getX() == 0 && pV->mConfidence > 0.9)
                            {
                                pHead->mpLabel->mLabelIndex = pV->mLabelIndex;

                                LOGD("Believe: %s", this->mpLabels[pV->mLabelIndex].getLabel());

                                if (gRedisContext)
                                {
                                    char id[3];
                                    sprintf(id, "%d", pV->mLabelIndex);

                                    redisCommand(gRedisContext, "PUBLISH %s %s", "tts", id);
                                }

                                this->mpLabels[pV->mLabelIndex].setX(1);
                                this->mpLabels[pV->mLabelIndex].setY(1);
                                //this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);
                                this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;
                                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;

                                this->mpLabels[labelIndex].setX(-1); // Clear
                            }
                            else
                            {
                                LOGD("One more try: %s", this->mpLabels[labelIndex].getLabel());
                                this->mpLabels[labelIndex].setX(0); // Give a more try;
                                this->mpLabels[labelIndex].mVersion = this->mVersion;
                            }
                        }
                        else
                        {
                            if (this->mpLabels[labelIndex].getX() == 0)
                            {
                                this->mpLabels[labelIndex].setX(1);
                                this->mpLabels[labelIndex].mVersion = this->mVersion;
                                LOGD("Dismiss: %s", this->mpLabels[pV->mLabelIndex].getLabel());
                            }
                            else
                            {
                                LOGD("Ignore: %s", this->mpLabels[pV->mLabelIndex].getLabel());
                            }
                        }

                        break;
                    }

                    pHead = pHead->mpNext;
                }

                if (pHead == NULL && pV->mConfidence > 0.9)
                {
                    LOGD("Find New User: %s", this->mpLabels[pV->mLabelIndex].getLabel());

                    LabelListItem *pItem = new LabelListItem();

                    if (pItem == NULL)
                    {
                        delete pV;

                        return -1;
                    }

                    pItem->mpLabel = new Label();

                    if (pItem->mpLabel == NULL)
                    {
                        delete pItem;

                        return -1;
                    }

                    Label *pNewLabel = pItem->mpLabel;

                    pNewLabel->mLabelIndex = pV->mLabelIndex;

                    this->mpLabels[pV->mLabelIndex].setX(1);
                    this->mpLabels[pV->mLabelIndex].setY(1);
                    this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
                    this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);

                    pItem->mpNext = this->mpActiveLabelList;
                    this->mpActiveLabelList->mpPrev = pItem;

                    this->mpActiveLabelList = pItem;

                    if (gRedisContext)
                    {
                        char id[3];
                        sprintf(id, "%d", pV->mLabelIndex);

                        redisCommand(gRedisContext, "PUBLISH %s %s", "tts", id);
                    }
                }
            }
        }
    }

#else

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
                            tryNext = TRUE;

                            if (this->mpLabels[i].mConfidence < pV->mConfidence)
                            {
                                LOGD("Next try: %s", this->mpLabels[i].getLabel());
                                this->mpLabels[i].setX(-1); // Give a more try;

                                break; // shortcut
                            }
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
#endif

    delete pV;

    return 0;
}
