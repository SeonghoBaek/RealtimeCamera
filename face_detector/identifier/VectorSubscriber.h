//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_VECTORSUBSCRIBER_H
#define REALTIMECAMERA_VECTORSUBSCRIBER_H

#include "Common.h"
#include "Thread.h"
#include "RedisIO.h"

class VectorSubscriber:public Thread, public IAsyncCallback {
private:
    RedisIO *mpRedis;
    AsyncRedisIO *mpAsyncRedis;

    char *mpServer;
    int mPort;
    char *mpChannel;

public:
    IMPLEMENT_THREAD(run());

    VectorSubscriber(const char *server, int port, const char *channel);

    virtual ~VectorSubscriber();

    void onMessage(const char*) override;

    void run();
};

#endif //REALTIMECAMERA_VECTORSUBSCRIBER_H
