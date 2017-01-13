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
#include <dirent.h>

#define CURRENT_NUM_LABEL 11
#define MAX_LABEL_LENGTH 80
#define LABEL_DIRECTORY "../face_register/input"

class Label
{
private:
    int     mCenterX;
    int     mCenterY;
    char    mLabelString[MAX_LABEL_LENGTH+1];

public:
    int     mLeft;
    int     mRight;
    int     mTop;
    int     mBottom;
    unsigned int mVersion;

    Label()
    {
        strcpy(mLabelString, "Unknown");
        mCenterX = -1;
        mCenterY = -1;
        mLeft = -1;
        mRight = -1;
        mTop  = -1;
        mBottom = -1;
        mVersion = 0;
    }

    void setX(int x)
    {
        this->mCenterX = x;
    }

    int  getX()
    {
        return this->mCenterX;
    }

    void setY(int y)
    {
        this->mCenterY = y;
    }

    int  getY()
    {
        return this->mCenterY;
    }

    void setLabel(const char* label)
    {
        memset(this->mLabelString, 0, MAX_LABEL_LENGTH+1);
        strcpy(this->mLabelString, label);
    }

    char* getLabel()
    {
        return this->mLabelString;
    }

    unsigned int setVersion(int version)
    {
        this->mVersion = version;
    }

    unsigned int versionUp()
    {
        this->mVersion++;
    }

    unsigned int getVersion()
    {
        return this->mVersion;
    }

    virtual ~Label() {}
};

class LabelListItem
{
public:
    Label*  mpLabel;
    LabelListItem *mpNext;
    LabelListItem *mpPrev;

    LabelListItem()
    {
        mpLabel = NULL;
        mpNext = NULL;
        mpPrev = NULL;
    }
};

class VectorClassFinder:public IVectorNotifier, public Thread, public ILooper
{
private:
    VectorQueue mVQ;
    Looper      *mpLooper;
    RedisIO     *mpRedisIO;
    int         mCurrentFrame;
    int         mNextFrame;
    Mutex_t     mFrameLock;
    int         mNumLabel;
    Label       *mpLabels;
    LabelListItem *mpActiveLabelList;
    unsigned int mVersion;

    float       getDistance(int sx, int sy, int tx, int ty);

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
        mNumLabel = 0; // TO DO: Load from file
        mpActiveLabelList = NULL;
        mpLabels = NULL;
        mVersion = 0;

        DIR *pDp;
        struct dirent *pDirent;
        pDp = opendir(LABEL_DIRECTORY);

        if (pDp)
        {
            int numLabels = 0;

            readdir(pDp); // .
            readdir(pDp); // ..

            while (pDirent = readdir(pDp))
            {
                numLabels++;
            }

            closedir(pDp);

            this->mNumLabel = numLabels;
        }

        pDp = opendir(LABEL_DIRECTORY);

        if (pDp && this->mNumLabel > 0)
        {
            this->mpLabels = new Label[this->mNumLabel];

            readdir(pDp);
            readdir(pDp);

            int i = 0;

            while (pDirent = readdir(pDp))
            {
                this->mpLabels[i].setLabel(pDirent->d_name);
                i++;
            }

            closedir(pDp);
        }

        for (int i = 0; i < this->mNumLabel; i++)
        {
            LOGD("LABEL: %s", this->mpLabels[i].getLabel());
        }
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

        if (mpLabels)
        {
            delete [] mpLabels;
        }
    }

    int nodtify(float data1, Vector& vector) override;

    int looperCallback(const char *event) override;

    char* getClosestLabel(int center_x, int center_y);

    char* getClosestIoULabel(int left, int right, int top, int bottom);

    void run();
};

#endif //REALTIMECAMERA_VECTORCLASSFINDER_H
