//
// Created by major on 17. 1. 6.
//

#ifndef REALTIMECAMERA_VECTORNETSUBSCRIBER_H
#define REALTIMECAMERA_VECTORNETSUBSCRIBER_H


#include <netinet/in.h>
#include "Thread.h"
#include "VectorQueue.h"
#include "VectorClassFinder.h"
#include "Lock.h"

#define SOCKET_TIME_OUT			1000 // mili
#define SOCK_PAGE_SIZE          4096

typedef struct ClientAddress
{
    char ipstr[INET6_ADDRSTRLEN+1];
    int  port;
} ClientAddress_t;

class VectorNetSubscriber: public Thread{
private:
    int     mSd;
    int     mClientSd;
    int     mPort;
    char    mIpStr[INET6_ADDRSTRLEN+1];
    char    mBuff[SOCK_PAGE_SIZE];
    IVectorNotifier *mpNotifier;
    Mutex_t mLock;

    int acceptOnSocket(int sd, struct ClientAddress* pClientAddr);
    int setupServerSocket(const char *address, int port);
    int receiveFromSocket(int sd, void* buffer, size_t bufferSize);
    int safeRead(void *buff, unsigned int length, int timeout);
    int sendToSocket(int sd, const void* buffer, size_t bufferSize);

public:
    IMPLEMENT_THREAD(this->run());

    VectorNetSubscriber(const char* ipString, int port)
    {
        if (ipString == NULL) // Local Host
        {
            sprintf(mIpStr, "%s", "127.0.0.1");
        }
        else
        {
            sprintf(mIpStr, "%s", ipString);
        }

        mPort = port;
        mSd = -1;
        mClientSd = -1;
        mpNotifier = NULL;
        mLock = Lock::createMutex();
    }

    VectorNetSubscriber(const char* ipString, int port, IVectorNotifier *pNotifier)
    {
        if (ipString == NULL) // Local Host
        {
            sprintf(mIpStr, "%s", "127.0.0.1");
        }
        else
        {
            sprintf(mIpStr, "%s", ipString);
        }

        mPort = port;
        mSd = -1;
        mClientSd = -1;
        mpNotifier = pNotifier;
        mLock = Lock::createMutex();
    }

    virtual ~VectorNetSubscriber()
    {
        if (mLock) Lock::deleteMutex(mLock);
    }

    void setVectorNotifier(IVectorNotifier* pNotifier)
    {
        LOCK(mLock)
        {
            this->mpNotifier = pNotifier;
        }
    }

    void run();
};


#endif //REALTIMECAMERA_VECTORNETSUBSCRIBER_H
