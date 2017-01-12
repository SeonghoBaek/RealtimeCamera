#include "VectorNetSubscriber.h"
#include "Common.h"
#include "Lock.h"
#include "Log.h"

//char *label_array[11] = {"SeonghoBaek", "ByongrakSeo", "HyungkiNoh", "kiyoungKim", "MinsamKo", "YonbeKim", "DaeyoungPark", "JangHyungLee", "KwangheeLee", "SanghoonLee", "Unknown"};

volatile int G_LABEL_INDEX = 10;

void VectorNetSubscriber::run()
{
    int clientSock = -1;

    LOCK(this->mLock)
    {
        if (this->mpNotifier == NULL)
        {
            LOGE("Set Vector Notifier");
            return;
        }
    }

    if ( (this->mSd = this->setupServerSocket(this->mIpStr, this->mPort)) == -1 )
    {
        LOGE("Server Setup Failure");
        return;
    }

    while (1)
    {
        clientSock = this->acceptOnSocket(this->mSd, NULL); // We accept only one client.

        if (clientSock != -1)
        {
            int length = -1;
            LOGI("Client Accepted");
            this->mClientSd = clientSock;

            while (1)
            {
                if ( (length = this->safeRead(this->mBuff, SOCK_PAGE_SIZE, -1)) < 0)
                {
                    LOGE("Socket Read Error");
                    break;
                }

                float *f_array = (float *)this->mBuff;

                int     left = (int)f_array[0];
                int     right = (int)f_array[1];
                int     top = (int)f_array[2];
                int     bottom = (int)f_array[3];
                float   confidence = f_array[4];
                int     label_index = (int)f_array[5];

                Vector v;
                int frameNum = -1;

                //v.mFrame = (int)(*(float *)(&(this->mBuff[0])));
                v.mX = left;
                v.mY = right;
                v.mT = top;
                v.mB = bottom;

                v.mConfidence = confidence;
                v.mLabelIndex = label_index;

                this->mpNotifier->nodtify(-1, v);
            }

            this->mClientSd = -1;
        }
    }

}

int VectorNetSubscriber::receiveFromSocket(int sd, void* buffer, size_t bufferSize)
{
    //ssize_t recvSize = recv(sd, buffer, bufferSize, 0);
    ssize_t recvSize = read(sd, buffer, bufferSize);
    // If receive is failed
    if (-1 == recvSize)
    {
        LOGE("Socket receive error: %s\n", strerror(errno));
    }

    return (int)recvSize;
}

int VectorNetSubscriber::safeRead(void *buff, unsigned int length, int timeout)
{
    int recv = 0;
    int MAX_TIMEOUT = -1; // 60 seconds
    int retry = 0;

    struct pollfd fds[1];

    fds[0].fd = this->mClientSd;
    fds[0].events = POLLIN | POLLERR | POLLHUP | POLLPRI;

    MAX_TIMEOUT = timeout;

    for(;;)
    {
        fds[0].revents = 0;

        poll(fds, 1, SOCKET_TIME_OUT);

        if (fds[0].revents & POLLIN || fds[0].revents & POLLPRI)
        {
            recv = (int)read(this->mClientSd, buff, length);
            //float *pF = (float *)buff;
            //LOGD("%d bytes, %f ", recv, pF[0]);
        }
        else if (fds[0].revents & POLLHUP)
        {
            LOGE("Connection Closed");

            recv = -2;
            break;
        }
        else if (fds[0].revents & POLLERR)
        {
            LOGE("Poll ERROR");

            recv = -2;
            break;
        }
        else if (MAX_TIMEOUT > 0)
        {
            bool bExit = FALSE;

            retry++;

            LOGE("Poll Time Out: %d", retry);

            if (retry >= MAX_TIMEOUT)
            {
                LOGE("Client Time Out!");
                bExit = TRUE;
            }

            if (bExit == TRUE)
            {
                recv = -2;

                break;
            }

            continue;
        }
        else
        {
            //LOGD("Waiting");
            continue;
        }

        if (recv <= 0)
        {
            strerror(errno);

            LOGE("Connection Closed: %s");
        }


        break;
    }

    return recv;
}

int VectorNetSubscriber::acceptOnSocket(int sd, struct ClientAddress* pClientAddr)
{
    struct sockaddr_storage client_addr;
    socklen_t clientLen = sizeof(client_addr);
    int clientSocket = accept(sd, (struct sockaddr *)&client_addr, &clientLen);

    if (clientSocket < 0)
    {
        LOGE("%s", strerror(errno));
        return -1;
    }

    if (pClientAddr)
    {
        struct sockaddr_in addr_in;

        addr_in = *(struct sockaddr_in *)&client_addr;

        inet_ntop(AF_INET, &addr_in.sin_addr, pClientAddr->ipstr, (socklen_t)sizeof(pClientAddr->ipstr));

        pClientAddr->port = ntohs(addr_in.sin_port);

        LOGI("Client Address: %s : %d", pClientAddr->ipstr, pClientAddr->port);
    }

    return clientSocket;
}

int VectorNetSubscriber::sendToSocket(int sd, const void* buffer, size_t bufferSize)
{
    ssize_t sentSize = 0;
    int length = (int)bufferSize;
    int nrPages = length / SOCK_PAGE_SIZE + 1;
    int sizeToSend = 0;

    //sentSize = write(sd, buffer, bufferSize);

    for (int i = 0; i < nrPages; i++)
    {
        if (length < SOCK_PAGE_SIZE)
        {
            sizeToSend = length;
        }
        else
        {
            sizeToSend = SOCK_PAGE_SIZE;
        }

        //send(sd, buffer + i*SOCK_PAGE_SIZE, sizeToSend, 0);
        int sent = 0;

        do
        {
            int w = (int)write(sd, (char *)buffer + sent + i*SOCK_PAGE_SIZE, sizeToSend);

            if (w <= 0)
            {
                //LOGE("Send Failure.");
                sent = 0;
                break;
            }

            sent += w;
            sizeToSend -= w;

            //LOGI("Socket Write Size: %d", w);
        }
        while(sizeToSend);

        length -= sent;
        sentSize += sent;
    }

    return (int)sentSize;
}

int VectorNetSubscriber::setupServerSocket(const char *address, int port)
{
    int sock = 0;
    int socketKeepAlive = TRUE;
    int socketReuseAddress = TRUE;

    struct sockaddr_in server_addr;

    memset(&server_addr, 0, sizeof(server_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    sock = socket(PF_INET, SOCK_STREAM, 0);

    if (sock < 0)
    {
        LOGE("%s", strerror(errno));
        return -1;
    }

    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &socketKeepAlive, sizeof socketKeepAlive);
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &socketReuseAddress, sizeof socketReuseAddress);

    if (strcmp(address, "*") != 0)
    {
        server_addr.sin_addr.s_addr = inet_addr(address);

        if (server_addr.sin_addr.s_addr == INADDR_NONE)
        {
            LOGE("Invalid server address.\n");
            return -1;
        }
    }
    else
    {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    }

    if (bind(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        LOGE("Server Socket Bind Error\n");
        return -1;
    }

    if (listen(sock, 10) < 0)
    {
        LOGE("Server Socket Listen Error\n");
        return -1;
    }

    return sock;
}
