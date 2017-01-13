//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_REDIS_IO_H
#define REALTIMECAMERA_REDIS_IO_H

#include "Common.h"
#include <hiredis.h>
#include <async.h>
#include <adapters/libevent.h>
#include <event2/event.h>
#include "Lock.h"
#include "Log.h"

#define MAX_HOST_STRING_LENGTH 80
#define MAX_CHANNEL_LENGTH 80
#define MAX_CMD_LENGTH 80

class IAsyncCallback {
public:
    virtual void onMessage(const char *) = 0;

    virtual ~IAsyncCallback() {}
};

class RedisIO {
protected:
    redisContext *mpRedisContext;
    char *mpHost;
    int mPort;
    Mutex_t mLock;

public:
    RedisIO();

    RedisIO(const char *host, int port);

    // timeout: seconds
    bool connect(int timeout);

    redisContext* getContext()
    {
        return this->mpRedisContext;
    }

    virtual ~RedisIO();
};

class AsyncRedisIO : public RedisIO {
private:
    struct event_base *mpEventBase;
    redisAsyncContext *mpRedisAsyncContext;
    char *mpChannel;

public:
    AsyncRedisIO();

    AsyncRedisIO(const char *host, int port, const char* channel);

    void subscribe(const char *pChannel, IAsyncCallback *pCallbackInstance);

    void unsubscribe();

    virtual ~AsyncRedisIO();
};

#endif //REALTIMECAMERA_REDIS_IO_H
