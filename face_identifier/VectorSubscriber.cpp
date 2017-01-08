//
// Created by major on 17. 1. 5.
//

#include "VectorSubscriber.h"

VectorSubscriber::VectorSubscriber(const char *server, int port, const char *channel)
{
    //pRedis->connect(1);
    if (server == NULL)
    {
        LOGE("Server is NULL");
    }

    if (channel == NULL)
    {
        LOGE("Channel is NULL");
    }

    this->mpServer = new char[MAX_HOST_STRING_LENGTH];
    sprintf(this->mpServer, "%s", server);

    this->mpChannel = new char[MAX_CHANNEL_LENGTH];
    sprintf(this->mpChannel, "%s", channel);

    this->mPort = port;

    this->mpAsyncRedis =new AsyncRedisIO(server, port, this->mpChannel);
}

VectorSubscriber::~VectorSubscriber()
{
    if (this->mpServer) delete [] this->mpServer;
    if (this->mpChannel) delete [] this->mpChannel;

    this->mpAsyncRedis->unsubscribe();
}

void VectorSubscriber::run()
{
    this->mpAsyncRedis->subscribe(this->mpChannel, this);
}

void VectorSubscriber::onMessage(const char* pMessage)
{
    char *pTemp = new char[strlen(pMessage)+1];
    strcpy(pTemp, pMessage);

    char *token;
    int i = 0;
    double vector[128];

    memset(vector, 0, sizeof(double) * 128);
    token = strtok(pTemp, "[]\n ");

    if (token)
    {
        vector[i] = atof(token);
        printf("%d %lf\n", i, vector[i]);
        i++;
    }

    while (token)
    {
        token = strtok(NULL, "[]\n ");
        if (token)
        {
            vector[i] = atof(token);
            //printf("%d %lf\n", i, vector[i]);
            i++;
        }
    }

    printf("%d\n", i);

    delete [] pTemp;
    //LOGI("MESSAGE: %s", pMessage);
}
