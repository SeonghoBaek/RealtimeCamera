#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "bridge.h"

static int bridge_receiveFromSocket(int sd, char* buffer, int size)
{
    ssize_t recvSize = recv(sd, buffer, size - 1, 0);
    // If receive is failed
    if (-1 == recvSize)
    {
        printf("Socket receive error\n");
    }
    else
    {
        // NULL terminate the buffer to make it a string
        buffer[recvSize] = 0;
    }

    return recvSize;
}

static int bridge_sendToDomain(const char *name, void* buff, int size)
{
    int localSocket = -1;

    if ((localSocket = socket(PF_LOCAL, SOCK_STREAM, 0)) < 0)
    {
        printf("NodeBus::sendLocalMessage: Local Socket Creation Error\n");

        return -1;
    }

    struct sockaddr_un address;
    const size_t nameLength = strlen(name);

    size_t pathLength = nameLength;

    int abstractNamespace = ('/' != name[0]);

    if (abstractNamespace)
    {
        pathLength++;
    }

    if (pathLength > sizeof(address.sun_path))
    {
        printf("NodeBus::sendLocalMessage: Socket Path Too Long\n");
        close(localSocket);

        return -1;
    }
    else
    {
        memset(&address, 0, sizeof(address));

        address.sun_family = PF_LOCAL;

        char* sunPath = address.sun_path;

        // First byte must be zero to use the abstract namespace
        if (abstractNamespace)
        {
            *sunPath++ = 0;
        }

        strcpy(sunPath, name);

        socklen_t addressLength = (offsetof(struct sockaddr_un, sun_path)) + pathLength;

        if (connect(localSocket, (struct sockaddr*) &address, addressLength) < 0)
        {
            printf("Bridge: Local Socket Connect Error\n");
            close(localSocket);

            return -1;
        }
    }

    send(localSocket, buff, size, 0);

    close(localSocket);

    return 0;
}

static int bridge_bindLocalSocketToName(int sd, const char* name)
{
    struct sockaddr_un address;

    const size_t nameLength = strlen(name);

    size_t pathLength = nameLength;

    // If name is not starting with a slash it is
    // in the abstract namespace
    int abstractNamespace = ('/' != name[0]);

    // Abstract namespace requires having the first
    // byte of the path to be the zero byte, update
    // the path length to include the zero byte
    if (abstractNamespace)
    {
        pathLength++;
    }

    if (pathLength > sizeof(address.sun_path))
    {
        printf("Socket Path Too Long\n");

        return -1;
    }
    else
    {
        memset(&address, 0, sizeof(address));

        address.sun_family = PF_LOCAL;

        char* sunPath = address.sun_path;

        // First byte must be zero to use the abstract namespace
        if (abstractNamespace)
        {
            *sunPath++ = 0;
        }

        strcpy(sunPath, name);

        socklen_t addressLength = (offsetof(struct sockaddr_un, sun_path)) + (socklen_t)pathLength;

        // Unlink if the socket name is already binded
        unlink(address.sun_path);

        if (bind(sd, (struct sockaddr*) &address, addressLength) < 0)
        {
            printf("Bridge Bind Error\n");

            return -1;
        }
    }

    return 0;
}

static int bridge_createSocket()
{
    int sd = -1;

    if ((sd = socket(PF_LOCAL, SOCK_STREAM, 0)) < 0)
    {
        printf("SocketBridge Local Socket Creation Error\n");
    }

    return sd;
}

static int bridge_thread(const char* domain_name, void (*action)(void *, int))
{
    int localSocket = -1;
    char buffer[BUFFER_SIZE];
    int __exit = 0;

    if ((localSocket = bridge_createSocket()) < 0)
    {
        printf("Local Socket Creation Error\n");

        return -1;
    }

    if (bridge_bindLocalSocketToName(localSocket, domain_name) < 0)
    {
        close(localSocket);

        return -1;
    }

    if (listen(localSocket, 4) < 0)
    {
        close(localSocket);

        return -1;
    }

    while (!__exit)
    {
        // Wait client.
        int clientSocket = accept(localSocket, NULL, NULL);

        if (clientSocket < 0)
        {
            printf("Accept Local Socket Error\n");

            __exit = 1;
        }

        ssize_t recvSize = bridge_receiveFromSocket(clientSocket, buffer, BUFFER_SIZE - 1);

        close(clientSocket);

        if (recvSize > 0)
        {
            if (strcmp(buffer, "EXIT") == 0) __exit = 1;

            action(buffer, (int)recvSize);
        }
    }

    close(localSocket);

    return 0;
}

static void* __thread_func(void *data)
{
    bridge_arg *t = (bridge_arg *)data;

    bridge_thread(t->name, t->cb);

    return NULL;
}

int bridge_create(char* my_name, void (*action)(void *, int))
{
    pthread_t tid;
    bridge_arg* arg;

    arg = (bridge_arg *)malloc(sizeof(bridge_arg));

    arg->name = my_name;
    arg->cb = action;

    if (pthread_create(&tid, NULL, __thread_func, arg) == 0)
    {
        printf("TID: %d\n", tid);
        pthread_detach(tid);
    }

    return tid;
}

int bridge_cross(const char* receiver_name, void *stream, int length)
{
    int stat = -1;

    stat = bridge_sendToDomain(receiver_name, stream, length);

    return stat;
}
