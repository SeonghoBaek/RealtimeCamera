//
// Created by major on 17. 1. 5.
//

#include "RedisIO.h"

RedisIO::RedisIO() : mpRedisContext(NULL), mPort(6379) {
    mLock = Lock::createMutex();
    this->mpHost = new char[MAX_HOST_STRING_LENGTH];
    sprintf(this->mpHost, "%s", "127.0.0.1");
}

RedisIO::RedisIO(const char *host, int port) : mpRedisContext(NULL) {
    this->mpHost = new char[MAX_HOST_STRING_LENGTH];
    sprintf(this->mpHost, "%s", host);
    this->mPort = port;
    mLock = Lock::createMutex();
}

bool RedisIO::connect(int second) {
    redisReply *reply;
    struct timeval timeout = {second, 0};

    LOCK(this->mLock) {
        if (mpRedisContext) return TRUE;

        mpRedisContext = redisConnectWithTimeout(this->mpHost, this->mPort, timeout);

        if (mpRedisContext == NULL || mpRedisContext->err) {
            if (mpRedisContext) {
                LOGE("Connection Error: %s\n", mpRedisContext->errstr);
                redisFree(mpRedisContext);
                mpRedisContext = NULL;

                return FALSE;
            } else {
                LOGE("Connection error: can't allocate redis context\n");

                return FALSE;
            }
        }
    }

    LOGI("Redis Connected: %s:%d", this->mpHost, this->mPort);

    return TRUE;
}

RedisIO::~RedisIO() {
    if (this->mpRedisContext) {
        LOGI("Free RedisContext");
        redisFree(this->mpRedisContext);
    }

    if (this->mLock) {
        Lock::deleteMutex(this->mLock);
    }

    if (this->mpHost) {
        delete[] this->mpHost;
    }
}

void redisAsyncCallbackHelper(redisAsyncContext *pContext, void *pReply, void *pPrivdata) {
    redisReply *pR = (redisReply *) pReply;

    if (pReply == NULL) return;

    if (pR->type == REDIS_REPLY_ARRAY) {
        for (int i = 0; i < pR->elements; i++) {
            //LOGI("%u) %s", i, pR->element[i]->str);
            IAsyncCallback *pInstanceCallback = (IAsyncCallback *) pPrivdata;
            if (pR->element[i]->str)
            {
                pInstanceCallback->onMessage(pR->element[i]->str);
            }
        }
    }
}

AsyncRedisIO::AsyncRedisIO() {
    RedisIO();
}

AsyncRedisIO::AsyncRedisIO(const char *host, int port, const char* channel) {
    RedisIO(host, port);
    this->mpChannel = new char[MAX_CHANNEL_LENGTH];
    sprintf(this->mpChannel, "%s", channel);
}

void AsyncRedisIO::subscribe(const char *pChannel, IAsyncCallback *pCallbackInstance) {
    struct event_base *base = event_base_new();
    LOGD("Connect to Redis");
    this->mpRedisAsyncContext = redisAsyncConnect(this->mpHost, this->mPort);

    redisLibeventAttach(this->mpRedisAsyncContext, base);
    char cmd[MAX_CMD_LENGTH];
    sprintf(cmd, "SUBSCRIBE %s", this->mpChannel);

    LOGD("send SUBSCRIBE %s", this->mpChannel);
    redisAsyncCommand(this->mpRedisAsyncContext, redisAsyncCallbackHelper, pCallbackInstance, cmd);
    LOGD("start dispatch");
    event_base_dispatch(base);
    LOGD("end dispatch");
}

void AsyncRedisIO::unsubscribe() {

}

AsyncRedisIO::~AsyncRedisIO() {
}