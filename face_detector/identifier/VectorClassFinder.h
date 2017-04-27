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
#define LABEL_DIRECTORY "../face_register/input/user"
#define LABEL_DIRECTORY_IGUEST "../face_register/input/iguest"
#define LABEL_DIRECTORY_OGUEST "../face_register/input/oguest"

#define LABEL_INVALIDATE_STATE -1
#define LABEL_READY_STATE -1
#define LABEL_VALID_STATE 1

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
    float   mConfidence;
    double  mUpdateTime;
    int     mChecked;
    int     mLabelIndex;

    Label()
    {
        strcpy(mLabelString, "Unknown");
        mCenterX = LABEL_INVALIDATE_STATE;
        mCenterY = -1;
        mLeft = 0;
        mRight = 0;
        mTop  = 0;
        mBottom = 0;
        mVersion = 0;
        mUpdateTime = 0;
        mChecked = 0;
        mLabelIndex = -1;
        mConfidence = 0.85;
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
        mpPrev = this;
    }

    virtual ~LabelListItem()
    {
        if (mpLabel)
        {
            delete mpLabel;
        }
    }
};

class VectorClassFinder:public IVectorNotifier, public Thread, public ILooper, public ITimer
{
private:
    VectorQueue     mVQ;
    Looper          *mpLooper;
    RedisIO         *mpRedisIO;
    int             mCurrentFrame;
    int             mNextFrame;
    Mutex_t         mFrameLock;
    Mutex_t         mBridgeLock;
    int             mNumLabel;
    Label           *mpLabels;
    LabelListItem   *mpActiveLabelList;
    unsigned int    mVersion;
    double          mLastBridgeSendTime;
    Timer           *mpTimer;

    float       getDistance(int sx, int sy, int tx, int ty);
    float       getIoU(int sleft, int sright, int stop, int sbottom, int left, int right, int top, int bottom);
    void        updateLabel(Label *pLabel, Vector* pVector);
    int         sendToBridge(const char *name, void* buff, int size);

public:
    IMPLEMENT_THREAD(run());

    VectorClassFinder()
    {
        mpLooper = new Looper(this);
        mpRedisIO = NULL;
        //mpRedisIO = new RedisIO();
        //mpRedisIO->connect(2);
        mCurrentFrame = -1;
        mNextFrame = -1;
        mFrameLock = Lock::createMutex();
        mBridgeLock = Lock::createMutex();
        mNumLabel = 0; // TO DO: Load from file
        mpActiveLabelList = NULL;
        mpLabels = NULL;
        mVersion = 0;
        mLastBridgeSendTime = 0;
        mpTimer = new Timer();

        this->mpTimer->setTimer(this, 3000);
    }

    virtual ~VectorClassFinder()
    {
        if (mFrameLock)
        {
            Lock::deleteMutex(this->mFrameLock);
        }

        if (mBridgeLock)
        {
            Lock::deleteMutex(this->mBridgeLock);
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

    void loadLabel()
    {
        DIR *pDp;
        struct dirent *pDirent;
        pDp = opendir(LABEL_DIRECTORY);
        const char* unknown = "Unknown";
        struct stat statbuf;
        char temp[80];

        if (mpLabels)
        {
            delete [] mpLabels;

            mpLabels = NULL;
        }

        if (pDp)
        {
            int numLabels = 0;

            readdir(pDp); // .
            readdir(pDp); // ..

            while (pDirent = readdir(pDp))
            {
                sprintf(temp, "%s/%s", LABEL_DIRECTORY, pDirent->d_name);
                stat(temp, &statbuf);

                if (S_ISDIR(statbuf.st_mode) && pDirent->d_name[0] != '.')
                {
                    if (strcmp(pDirent->d_name, unknown)) {
                        numLabels++;
                    }
                }
            }

            closedir(pDp);

            this->mNumLabel = numLabels;
        }

        pDp = opendir(LABEL_DIRECTORY_IGUEST);

        if (pDp)
        {
            int numLabels = 0;

            readdir(pDp); // .
            readdir(pDp); // ..

            while (pDirent = readdir(pDp))
            {
                sprintf(temp, "%s/%s", LABEL_DIRECTORY, pDirent->d_name);
                stat(temp, &statbuf);

                if (S_ISDIR(statbuf.st_mode) && pDirent->d_name[0] != '.')
                {
                    if (strcmp(pDirent->d_name, unknown)) {
                        numLabels++;
                    }
                }
            }

            closedir(pDp);

            this->mNumLabel += numLabels;
        }

        pDp = opendir(LABEL_DIRECTORY_OGUEST);

        if (pDp)
        {
            int numLabels = 0;

            readdir(pDp); // .
            readdir(pDp); // ..

            while (pDirent = readdir(pDp))
            {
                sprintf(temp, "%s/%s", LABEL_DIRECTORY, pDirent->d_name);
                stat(temp, &statbuf);

                if (S_ISDIR(statbuf.st_mode) && pDirent->d_name[0] != '.')
                {
                    if (strcmp(pDirent->d_name, unknown)) {
                        numLabels++;
                    }
                }
            }

            closedir(pDp);

            this->mNumLabel += numLabels;
        }

        this->mpLabels = new Label[this->mNumLabel];

        struct dirent **namelist;
        int n;

        n = scandir(LABEL_DIRECTORY, &namelist, 0, alphasort);
        int label_index = 0;

        if (n < 0)
            perror("scandir");
        else
        {
            for (int i = 2; i < n; i++)
            {
                sprintf(temp, "%s/%s", LABEL_DIRECTORY, namelist[i]->d_name);
                stat(temp, &statbuf);

                if (S_ISDIR(statbuf.st_mode) && namelist[i]->d_name[0] != '.')
                {
                    //printf("%s\n", namelist[i]->d_name);
                    if (strcmp(namelist[i]->d_name, unknown))
                    {
                        this->mpLabels[label_index].setLabel(namelist[i]->d_name);
                        label_index++;
                    }
                }

                free(namelist[i]);
            }

            free(namelist);
        }

        n = scandir(LABEL_DIRECTORY_IGUEST, &namelist, 0, alphasort);

        if (n < 0)
            perror("scandir");
        else
        {
            for (int i = 2; i < n; i++)
            {
                sprintf(temp, "%s/%s", LABEL_DIRECTORY_IGUEST, namelist[i]->d_name);
                stat(temp, &statbuf);

                if (S_ISDIR(statbuf.st_mode) && namelist[i]->d_name[0] != '.')
                {
                    //printf("%s\n", namelist[i]->d_name);
                    if (strcmp(namelist[i]->d_name, unknown))
                    {
                        this->mpLabels[label_index].setLabel(namelist[i]->d_name);
                        label_index++;
                    }
                }

                free(namelist[i]);
            }

            free(namelist);
        }

        n = scandir(LABEL_DIRECTORY_OGUEST, &namelist, 0, alphasort);

        if (n < 0)
            perror("scandir");
        else
        {
            for (int i = 2; i < n; i++)
            {
                sprintf(temp, "%s/%s", LABEL_DIRECTORY_OGUEST, namelist[i]->d_name);
                stat(temp, &statbuf);

                if (S_ISDIR(statbuf.st_mode) && namelist[i]->d_name[0] != '.')
                {
                    //printf("%s\n", namelist[i]->d_name);
                    if (strcmp(namelist[i]->d_name, unknown))
                    {
                        this->mpLabels[label_index].setLabel(namelist[i]->d_name);
                        label_index++;
                    }
                }

                free(namelist[i]);
            }

            free(namelist);
        }

        for (int i = 0; i < this->mNumLabel; i++)
        {
            LOGD("LABEL: %s", this->mpLabels[i].getLabel());
        }
    }

    int nodtify(float data1, Vector& vector) override;

    int looperCallback(const char *event) override;

    char* getClosestLabel(int center_x, int center_y) {return NULL;}

    char* getClosestIoULabel(int left, int right, int top, int bottom);

    void resetLabelCheck();

    int fireUserEvent(int labelIndex);

    int addNewFace(Vector *pV);

    void versionUp()
    {
        //LOGD("Version Up: %d", this->mVersion);

        LOCK(mFrameLock)
        {
            this->mVersion++;
        }
    }

    void run();

    int timerCallback(const char *event) override;
};

#endif //REALTIMECAMERA_VECTORCLASSFINDER_H
