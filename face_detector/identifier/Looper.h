//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_LOOPER_H
#define REALTIMECAMERA_LOOPER_H

#include "Common.h"

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

#endif //REALTIMECAMERA_LOOPER_H
