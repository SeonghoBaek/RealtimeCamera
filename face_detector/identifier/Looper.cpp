//
// Created by major on 17. 1. 5.
//

#include "Looper.h"
#include "Log.h"

int debugLooperCallback(const char *event)
{
    LOGE("fd: %d, events: %d\n", event);

    return 0;
}

Looper::Looper()
{
    this->mUser_cb = NULL;
    this->mpImpl = NULL;

    pipe(this->mFd);

    this->mTimeOut = -1;

    LOGI("PIPE: %d %d", this->mFd[0], this->mFd[1]);
}

Looper::Looper(LooperCallback_t func)
{
    this->mpImpl = NULL;

    pipe(this->mFd);

    this->mUser_cb = func;

    this->mTimeOut = -1;
}

Looper::Looper(ILooper* pImpl)
{
    this->mUser_cb = NULL;

    pipe(this->mFd);

    this->mpImpl = pImpl;

    this->mTimeOut = -1;
}

int Looper::timer()
{
    bool exit = FALSE;
    int res;
    char buff[BUFSIZ + 1];
    struct pollfd fds[1];

    fds[0].fd = this->mFd[0];
    fds[0].events = POLLIN;

    int mili = -1;

    for (;;)
    {
        fds[0].revents = 0;
        mili = this->mTimeOut;

        if (mili == -1)
        {
            LOGW("Looper time out is -1. INFINITE WAIT");
        }

        poll(fds, 1, mili);

        if (fds[0].revents & POLLIN)
        {
            res = 0;
            int nReadBytes = (int)read(fds[0].fd, buff, BUFSIZ);
            int nMsg = nReadBytes/LOOPER_EVT_LENGTH;

            //LOGI("LOOPER: %s, %d", buff, nMsg);

            for (int i = 0; i < nMsg; i++)
            {
                if (this->mpImpl)
                {
                    this->mpImpl->looperCallback(buff);
                }

                if (this->mUser_cb)
                {
                    this->mUser_cb(buff);
                }

                if (strncmp((char *)&buff[i*LOOPER_EVT_LENGTH], LOOPER_EXIT, LOOPER_EVT_LENGTH) == 0)
                {
                    exit = TRUE;
                    break;
                }
            }

            if (exit == TRUE) break;

            memset(buff, 0, sizeof(buff));
        }
        else
        {
            if (this->mpImpl)
            {
                this->mpImpl->looperCallback(LOOPER_TIMEOUT);
            }

            if (this->mUser_cb)
            {
                this->mUser_cb(LOOPER_TIMEOUT);
            }
        }
    }

    return res;
}

// -1: infinite
int Looper::wait(int mili)
{
    bool exit = FALSE;
    int res = 0;
    char buff[BUFSIZ + 1];
    struct pollfd fds[1];

    fds[0].fd = this->mFd[0];
    fds[0].events = POLLIN;

    for (;;)
    {
        fds[0].revents = 0;

        poll(fds, 1, mili);

        if (fds[0].revents & POLLIN)
        {
            int nReadBytes = (int)read(fds[0].fd, buff, BUFSIZ);
            int nMsg = nReadBytes/LOOPER_EVT_LENGTH;

            //LOGI("LOOPER: %s, %d", buff, nMsg);

            for (int i = 0; i < nMsg; i++)
            {
                if (this->mpImpl)
                {
                    this->mpImpl->looperCallback(buff);
                }

                if (this->mUser_cb)
                {
                    this->mUser_cb(buff);
                }

                if (strncmp((char *)&buff[i*LOOPER_EVT_LENGTH], LOOPER_EXIT, LOOPER_EVT_LENGTH) == 0)
                {
                    exit = TRUE;
                    break;
                }
            }

            if (exit == TRUE) break;

            memset(buff, 0, sizeof(buff));
        }
        else
        {
            break;
        }
    }

    return res;
}

int Looper::wakeup()
{
    write(this->mFd[1], LOOPER_WAKEUP, LOOPER_EVT_LENGTH);

    return 0;
}

int Looper::exit()
{
    write(this->mFd[1], LOOPER_EXIT, LOOPER_EVT_LENGTH);

    return 0;
}

void Looper::exit(int fd)
{
    write(fd, LOOPER_EXIT, LOOPER_EVT_LENGTH);
}

int Looper::sendMessage(const char* msg, int length)
{
    return (int)write(this->mFd[1], msg, length);
}

