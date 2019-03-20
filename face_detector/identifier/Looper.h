//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_LOOPER_H
#define REALTIMECAMERA_LOOPER_H

#include "Common.h"
#include "Thread.h"
#include "Lock.h"

#define LOOPER_EVT_LENGTH 4
#define LOOPER_WAKEUP 	"wake"
#define LOOPER_EXIT		"exit"
#define LOOPER_TIMEOUT	"tout"

typedef int (*LooperCallback_t)(const char* event);

class ILooper
{
public:
    virtual int looperCallback(const char* event) = 0;
    virtual ~ILooper() {}
};

class Looper
{
private:
    LooperCallback_t mUser_cb;
    ILooper* mpImpl;
    int mFd[2];
    int mTimeOut;

public:
    Looper();

    Looper(LooperCallback_t func);

    Looper(ILooper* pImpl);

    virtual ~Looper() {close(mFd[0]);close(mFd[1]);}

    // -1: infinite
    int wait(int mili);

    int timer();

    int wakeup();

    int exit();

    int sendMessage(const char* msg, int length);

    int getId() {return mFd[1];}

    int setTimeOut(int mili)
    {
        this->mTimeOut = mili;
        return mili;
    }

    static void exit(int id);
};

class ITimer
{
public:
    virtual int timerCallback(const char* event) = 0;
    virtual ~ITimer() {}
};

class TimerThread : public Thread, public ILooper
{
private:
    int     mWaitTime;
    ITimer  *mpTimerCallback;
    Looper  *mpLooper;

public:
    IMPLEMENT_THREAD(this->run())

    TimerThread(int mili, ITimer *pCallback)
    {
        this->mWaitTime = mili;
        this->mpTimerCallback = pCallback;
        this->mpLooper = new Looper(this);;
    }

    void run()
    {
        this->mpLooper->setTimeOut(this->mWaitTime);
        this->mpLooper->timer();
    }

    int looperCallback(const char *event) override
    {
        if (mpTimerCallback != NULL)
        {
            mpTimerCallback->timerCallback("timer");
        }

        return 0;
    }

    virtual ~TimerThread()
    {
        if (this->mpLooper)
        {
            this->mpLooper->exit();
        }

        delete this->mpLooper;
    }
};

class Timer : public ITimer
{
private:
    TimerThread *mTT;
    ITimer      *mpTimerCallback;
    Mutex_t     mLock;

public:

    Timer()
    {
        this->mTT = NULL;
        this->mpTimerCallback = NULL;
        this->mLock = Lock::createMutex();
    }

    int setTimer(ITimer *pCallback, int mili)
    {
        LOCK(this->mLock)
        {
            if (this->mpTimerCallback != NULL)
            {
                LOGD("Previous Timer Still Alive");
                return -1;
            }

            this->mpTimerCallback = pCallback;

            this->mTT = new TimerThread(mili, this);

            this->mTT->startThread();
        }

        return 0;
    }

    int timerCallback(const char* event)
    {
        LOCK(this->mLock)
        {
            if (this->mpTimerCallback != NULL)
            {
                this->mpTimerCallback->timerCallback(event);
            }
        }
    }

    virtual ~Timer()
    {
        if (this->mLock)
        {
            Lock::deleteMutex(this->mLock);
        }

        if (this->mTT)
        {
            delete this->mTT;
        }
    }
};

#endif //REALTIMECAMERA_LOOPER_H
