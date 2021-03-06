//
// Created by major on 17. 1. 6.
//

#include "VectorClassFinder.h"
#include "Log.h"
#include <math.h>
#include <hiredis.h>
#include "../sparse/sparse.h"
#include <ctime>
#include "identifier.h"

using namespace std;

#define RESET_FREQ 120
#define USE_UPDATE_CHECK 0
#define USE_LINED_LIST 1
#define IOU_SINGLE_MODE 0
#define BRIDGE_INTERVAL 6 // Sec.
#define USE_ROBOT 1
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define USE_TRACKING 1
#define LABEL_VERSION_DIFF 7

float CONFIDENCE_LEVEL = 0.97;
float CONFIDENCE_MIN_LEVEL = 0.90;

#if (IOU_SINGLE_MODE == 1)
float IOU_INTERSECT_NEW_USER = 0.7;
float IOU_INTERSECT_CUR_USER = 0.4;
float IOU_INTERSECT_TRACKING = 0.4;
#else
float IOU_INTERSECT_NEW_USER = 0.5;
float IOU_INTERSECT_CUR_USER = 0.4;
float IOU_INTERSECT_TRACKING = 0.7;
#endif

#if (USE_ROBOT == 1)
int gSupportRobot = 1;
#else
int gSupportRobot = 0;
#endif

redisContext    *gTTSRedisContext = NULL;

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

struct label_struct_t* VectorClassFinder::getLabelList()
{
    LOCK(this->mFrameLock)
    {
        LabelListItem *pItem = this->mpActiveLabelList;
        struct label_struct_t *p_last = NULL;
        struct label_struct_t *p_first = NULL;

        while (pItem != NULL)
        {
            if (abs((int)(this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion - this->mVersion)) < LABEL_VERSION_DIFF)
            {
                struct label_struct_t *p_label = (struct label_struct_t *) malloc(sizeof(struct label_struct_t));
                int labelIndex = pItem->mpLabel->mLabelIndex;
                p_label->top = this->mpLabels[labelIndex].mTop;
                p_label->bottom = this->mpLabels[labelIndex].mBottom;
                p_label->left = this->mpLabels[labelIndex].mLeft;
                p_label->right = this->mpLabels[labelIndex].mRight;
                strcpy(p_label->label, this->mpLabels[labelIndex].getLabel());
                p_label->p_next = NULL;

                if (p_last == NULL) {
                    p_last = p_label;
                    p_first = p_last;
                } else {
                    p_last->p_next = p_label;
                    p_last = p_label;
                }
            }

            pItem = pItem->mpNext;
        }

        return p_first;
    }

    return NULL;
}

char* VectorClassFinder::getClosestIoULabel(int left, int right, int top, int bottom)
{
    int   max_iou_index = -1;
    float max_iou = 0.0;
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

        this->mIgnoreInvalidation = 1; /* Added by Seongho Baek 2019.03.20 */

        while (pItem != NULL)
        {
            /*
             * Added by Seongho Baek 2019.03.20
             */
            if (this->mpLabels[pItem->mpLabel->mLabelIndex].getX() == LABEL_VALID_STATE)
            {
                int labelIndex = pItem->mpLabel->mLabelIndex;

                float iou = this->getIoU(left, right, top, bottom,
                                         this->mpLabels[labelIndex].mLeft,  this->mpLabels[labelIndex].mRight,
                                         this->mpLabels[labelIndex].mTop, this->mpLabels[labelIndex].mBottom);

                //LOGD("iou: %f", iou);

                if (iou > max_iou)
                {
                    max_iou_index = labelIndex;
                    max_iou = iou;
                    //IOU = iou;

                    //LOGD("OK Still %s", this->mpLabels[index].getLabel());
                }

                /* Commented out by Seongho Baek 2019.03.20
                //LOGD("Version: %d", abs(this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion - this->mVersion));
                int invalidate = 0;
                int tooOld = 0;

                if (this->mpLabels[pItem->mpLabel->mLabelIndex].getX() == LABEL_INVALIDATE_STATE)
                {
                    invalidate = 1;
                    LOGD("Invalidate: %s", this->mpLabels[pItem->mpLabel->mLabelIndex].getLabel());
                }
                */

                /*
                if (abs(this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion - this->mVersion) > RESET_FREQ)
                {
                    LOGD("Version: %d", abs(this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion - this->mVersion));
                    tooOld = 1;
                }

                if (invalidate == 1 || tooOld == 1)
                {
                    this->mpLabels[pItem->mpLabel->mLabelIndex].setX(LABEL_INVALIDATE_STATE);
                    this->mpLabels[pItem->mpLabel->mLabelIndex].mConfidence = 0.85;

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
                */
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

                if (abs((int)(this->mpLabels[i].mVersion - this->mVersion)) > 15)
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

    if (max_iou < IOU)
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

        LOGD("By IoU Tracking: %s", this->mpLabels[max_iou_index].getLabel());

        #if (USE_UPDATE_CHECK == 1)
        LOGD("latency: %f for %s", (float)(cur - this->mpLabels[max_iou_index].mUpdateTime), this->mpLabels[max_iou_index].getLabel());
        #endif
    }

    return this->mpLabels[max_iou_index].getLabel();
}

int VectorClassFinder::nodtify(float data1, Vector& vector)
{
    if (data1 == 1)
    {
        LOCK(this->mFrameLock)
        {
            this->loadLabel();
        }

        return 0;
    }

    this->mVQ.push(vector);

    this->mpLooper->wakeup();

    return 0;
}

void VectorClassFinder::run()
{

    const char *hostname = "10.144.164.202";
    int port = 8090; //6379;
    struct timeval timeout = {1, 500000};

    gTTSRedisContext = redisConnectWithTimeout(hostname, port, timeout);

    if (gTTSRedisContext == NULL || gTTSRedisContext->err)
    {
        if (gTTSRedisContext)
        {
            LOGE("Connection Error: %s, %s:%d\n", gTTSRedisContext->errstr, hostname, port);
            redisFree(gTTSRedisContext);
            gTTSRedisContext = NULL;
        } else {
            LOGE("Connection error: can't allocate redis context\n");
        }
    }

    this->mpLooper->wait(-1);
}

int VectorClassFinder::sendToBridge(const char *name, void* buff, int size)
{
    int     localSocket = -1;
    //struct  timeval time;
    //double  cur;

    /*
    if (gettimeofday(&time,NULL))
    {
        cur =  0;
    }
    else
    {
        cur = (double)time.tv_sec + (double)time.tv_usec * .000001;
    }

    if (this->mLastBridgeSendTime == 0)
    {
        this->mLastBridgeSendTime = cur;
    }
    else if (cur - this->mLastBridgeSendTime < BRIDGE_INTERVAL)
    {
        LOGI("Too Short Brigde Time Interval: %d\n", (int)(cur - this->mLastBridgeSendTime));
        return -1;
    }

    LOGI("Brigde Time Interval: %d\n", (int)(cur - this->mLastBridgeSendTime));

    this->mLastBridgeSendTime = cur;
    */

    if ((localSocket = socket(PF_LOCAL, SOCK_STREAM, 0)) < 0)
    {
        LOGE("Local Socket Creation Error\n");

        return -1;
    }

    struct sockaddr_un address;
    const size_t nameLength = strlen(name);

    size_t pathLength = nameLength;

    int abstractNamespace = ('/' != name[0]);

    if (abstractNamespace)
    {
        pathLength++;
    }

    if (pathLength > sizeof(address.sun_path))
    {
        LOGE("Socket Path Too Long\n");
        close(localSocket);

        return -1;
    }
    else
    {
        memset(&address, 0, sizeof(address));

        address.sun_family = PF_LOCAL;

        char* sunPath = address.sun_path;

        // First byte must be zero to use the abstract namespace
        if (abstractNamespace)
        {
            *sunPath++ = 0;
        }

        strcpy(sunPath, name);

        socklen_t addressLength = (offsetof(struct sockaddr_un, sun_path)) + pathLength;

        if (connect(localSocket, (struct sockaddr*) &address, addressLength) < 0)
        {
            LOGE("Local Socket Connect Error\n");
            close(localSocket);

            return -1;
        }
    }

    send(localSocket, buff, size, 0);

    close(localSocket);

    return 0;
}

int VectorClassFinder::fireUserEvent(int labelIndex)
{
    if (gSupportRobot)
    {
        time_t curr_time;
        struct tm *curr_tm;
        struct  timeval _time;
        double  cur;

        curr_time = time(NULL);
        curr_tm = localtime(&curr_time);

        if (gettimeofday(&_time,NULL))
        {
            cur =  0;
        }
        else
        {
            cur = (double)_time.tv_sec + (double)_time.tv_usec * .000001;
        }

        LOCK(this->mBridgeLock)
        {
            if (this->mLastBridgeSendTime == 0)
            {
                this->mLastBridgeSendTime = cur;
            }
            else if (cur - this->mLastBridgeSendTime < BRIDGE_INTERVAL)
            {
                LOGI("Too Short Brigde Time Interval: %d\n", (int) (cur - this->mLastBridgeSendTime));
                return -1;
            }

            //LOGI("Brigde Time Interval: %d\n", (int)(cur - this->mLastBridgeSendTime));

            this->mLastBridgeSendTime = cur;
        }

        if (curr_tm->tm_hour < 8 || curr_tm->tm_hour > 17)
        {
            LOGI("Do not open door for sequrity\n");

            if (gTTSRedisContext && labelIndex > -1)
            {
                char *name = "warning";
                redisCommand(gTTSRedisContext, "PUBLISH %s %s", "tts", name);
            }

            return 0;
        }

        if (curr_tm->tm_wday == 6 || curr_tm->tm_wday == 0)
        {
            LOGI("Do not open door for security\n");

            if (gTTSRedisContext && labelIndex > -1)
            {
                char *name = "warning";
                redisCommand(gTTSRedisContext, "PUBLISH %s %s", "tts", name);
            }

            return 0;
        }

		/*
        if (gTTSRedisContext && labelIndex > -1)
        {
            char *name = this->mpLabels[labelIndex].getLabel();
            redisCommand(gTTSRedisContext, "PUBLISH %s %s", "tts", name);
        }
		*/

		/*
        if (this->sendToBridge("/var/tmp/robot_bridge", (void *)"open", 4) < 0)
        {
            // Successful open.
            //this->mVersion = 0; // Reset.
            LOGI("Robot send error\n");
        }
		*/
    }
    else
    {
		/*
        if (gTTSRedisContext && labelIndex > -1)
        {
            char *name = this->mpLabels[labelIndex].getLabel();
            redisCommand(gTTSRedisContext, "PUBLISH %s %s", "tts", name);
        }
		*/
    }

    return 0;
}

int VectorClassFinder::addNewFace(Vector *pV)
{
    LOGD("Add New Face: %s",  this->mpLabels[pV->mLabelIndex].getLabel());

    if (pV == NULL) return -1;

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

    //Label *pNewLabel = pItem->mpLabel;

    pItem->mpLabel->mLabelIndex = pV->mLabelIndex;
    this->mpLabels[pV->mLabelIndex].setX(LABEL_VALID_STATE);
    this->mpLabels[pV->mLabelIndex].setY(1);
    this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
    this->mpLabels[pV->mLabelIndex].mLeft = pV->mX;
    this->mpLabels[pV->mLabelIndex].mRight = pV->mY;
    this->mpLabels[pV->mLabelIndex].mTop = pV->mT;
    this->mpLabels[pV->mLabelIndex].mBottom = pV->mB;

    //this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);

    if (this->mpActiveLabelList == NULL)
    {
        this->mpActiveLabelList = pItem;
    }
    else
    {
        pItem->mpNext = this->mpActiveLabelList;
        this->mpActiveLabelList->mpPrev = pItem;

        this->mpActiveLabelList = pItem;
    }

    return 0;
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
        if (pV->mConfidence < CONFIDENCE_MIN_LEVEL)
        {
            LOGD("Confidence(%f) low, %s, Reset to state %d\n", pV->mConfidence, this->mpLabels[pV->mLabelIndex].getLabel(), LABEL_INVALIDATE_STATE);

            this->mpLabels[pV->mLabelIndex].setX(LABEL_INVALIDATE_STATE);
            this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
            this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;

            /*
            if (this->mpActiveLabelList == NULL)
            {
                if (gTTSRedisContext)
                {
                    char *name = "default";
                    redisCommand(gTTSRedisContext, "PUBLISH %s %s", "tts", name);
                }
            }
            */

            delete pV;

            return 0;
        }

        int prevVersion = this->mpLabels[pV->mLabelIndex].mVersion;

        if (abs((int)(this->mVersion - prevVersion)) > LABEL_VERSION_DIFF) // Too old from first identification.
        {
            this->mpLabels[pV->mLabelIndex].setX(LABEL_INVALIDATE_STATE);
            LOGD("Too old. VerDiff:%d, %s label state: %d, Confidence: %f\n",
                    abs((int)(this->mVersion - prevVersion)),
                    this->mpLabels[pV->mLabelIndex].getLabel(),
                    this->mpLabels[pV->mLabelIndex].getX(),  pV->mConfidence);
        } else {
            LOGD("Label State %d, %s, Confidence: %f\n", this->mpLabels[pV->mLabelIndex].getX(),
                 this->mpLabels[pV->mLabelIndex].getLabel(), pV->mConfidence);
        }

        if (this->mpLabels[pV->mLabelIndex].getX() < LABEL_READY_STATE)
        {
            /*
             * example)
             * INVALIDATE_STATE = -3
             * READY_STATE = -1
             * VALID_STATE = 1
             * To be registered as a new identified face, state should be valid state.
             * From INVALIDATE_STATE, we need number of [READY_STATE - INVALIDATE_STATE] identified faces.
             */
            if (this->mpLabels[pV->mLabelIndex].getX() == LABEL_INVALIDATE_STATE)
            {
                if (pV->mConfidence > CONFIDENCE_LEVEL)
                {
                    this->mpLabels[pV->mLabelIndex].setX(this->mpLabels[pV->mLabelIndex].getX() + 1); // STATE Version Up
                    this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
                }
            } else {
                this->mpLabels[pV->mLabelIndex].setX(this->mpLabels[pV->mLabelIndex].getX() + 1); // STATE Version Up
                this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;
            }
        } else if (this->mpLabels[pV->mLabelIndex].getX() == LABEL_READY_STATE) {
            LOGD("Find New Face: %s", this->mpLabels[pV->mLabelIndex].getLabel());

            this->addNewFace(pV); // Now label is VALID_STATE
            this->fireUserEvent(pV->mLabelIndex);
        } else { // LABEL_VALID_STATE
            this->mpLabels[pV->mLabelIndex].setX(LABEL_VALID_STATE); // For explicit action
            this->mpLabels[pV->mLabelIndex].mVersion = this->mVersion;

            this->updateLabel(&this->mpLabels[pV->mLabelIndex], pV);

            this->mpLabels[pV->mLabelIndex].mConfidence = pV->mConfidence;
            //LOGD("Update %s confidence %f", this->mpLabels[pV->mLabelIndex].getLabel(), pV->mConfidence);

            #if (USE_UPDATE_CHECK == 1)
            this->mpLabels[pV->mLabelIndex].mUpdateTime = cur;
            #endif
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

                if (tryNext == FALSE)
                {
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
        }*/
    }
#endif

    delete pV;

    return 0;
}

int VectorClassFinder::invalidate(int force)
{
    int threshold = LABEL_VERSION_DIFF;

    if (force)
    {
        threshold = -1;
    }

    LOCK(this->mFrameLock)
    {
        LabelListItem *pItem = this->mpActiveLabelList;

        if (this->mIgnoreInvalidation) return -1; /* Added by Seongho Baek 2019.03.20 */

        //LOGD("Invalidate old label information.");

        while (pItem != NULL)
        {
            if (abs((int)(this->mpLabels[pItem->mpLabel->mLabelIndex].mVersion - this->mVersion)) > threshold)
            {
                if (force)
                {
                    LOGD("Invalidate by force\n");
                }

                this->mpLabels[pItem->mpLabel->mLabelIndex].setX(LABEL_INVALIDATE_STATE);
                this->mpLabels[pItem->mpLabel->mLabelIndex].mConfidence = 0.85;

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

            if (pItem != NULL) pItem = pItem->mpNext;
        }
    }

    return 0;
}

int VectorClassFinder::timerCallback(const char *event)
{
    /*
     * Added by Seongho Baek 2019.03.20
     *
     * Periodic label invalidation using frame version.
     * Invalidate old label.
     */
    if (this->invalidate(0) == -1)
    {
        LOCK(this->mFrameLock)
        {
            this->mIgnoreInvalidation = 0;
        }
    }

    return 0;

    /* Comment out by Seongho Baek 2019.03.20
    LOCK(this->mFrameLock)
    {
        //LOGD("Reset Timer Callback!\n");
        //this->mVersion = 0; // Reset.
        LabelListItem *pItem = this->mpActiveLabelList;

        while (pItem != NULL)
        {
            this->mpLabels[pItem->mpLabel->mLabelIndex].setX(LABEL_INVALIDATE_STATE);

            pItem = pItem->mpNext;
        }
    }

    return 0;
    */
}
