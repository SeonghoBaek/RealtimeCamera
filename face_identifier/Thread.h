//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_THREAD_H
#define REALTIMECAMERA_THREAD_H

#include "Log.h"
#include "Common.h"

#define THREAD_NAME_LENGTH 64

typedef pthread_t THREAD_ID_T;

// Should be thread safe.
void* _thread_func(void *);

typedef void* (*ThreadRun_t)(void *);

class ThreadHelper
{
public:
    static void run(ThreadRun_t thread_run, void *arg, bool bDetached)
    {
        THREAD_ID_T tid;

        if (pthread_create(&tid, NULL, thread_run, arg) == 0)
        {
            if (bDetached == TRUE)
            {
                pthread_detach(tid);
            }
            else
            {
                void *res;

                pthread_join(tid, &res);
            }
        }
    }
};

class Thread
{
private:
    THREAD_ID_T	tid;
    char mThreadName[THREAD_NAME_LENGTH+1];

    static void *activateThread(void* instance)
    {
        ((Thread *)instance)->loop();

        return NULL;
    }

protected:
    bool 				detached;

public:
    Thread():detached(TRUE), tid((THREAD_ID_T)-1) {mThreadName[0] = '\0';}

    Thread(bool d):detached(d), tid((THREAD_ID_T)-1) {mThreadName[0] = '\0';}

    void startThread()
    {
        if (pthread_create(&tid, NULL, activateThread, this) == 0)
        {
            if (this->detached == TRUE)
            {
                //LOGI("Start Thread, Object: %s, ID: %u\n", mThreadName, tid);

                pthread_detach(tid);
            }
            else
            {
                void *res;

                pthread_join(tid, &res);
            }
        }
    }

    void setDetach(bool bDetached)
    {
        this->detached = bDetached;
    }

    void setThreadName(char *name)
    {
        memset(mThreadName, 0, THREAD_NAME_LENGTH+1);
        strncpy(mThreadName, name, THREAD_NAME_LENGTH);
    }

    THREAD_ID_T getThreadID(){return this->tid;}

    virtual ~Thread() {}

    virtual int loop() = 0;
};

#define IMPLEMENT_THREAD(_func) 		\
		virtual int loop() 	{ _func; return 0; } \

#endif //REALTIMECAMERA_THREAD_H
