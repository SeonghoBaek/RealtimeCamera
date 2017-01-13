//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_COMMON_H
#define REALTIMECAMERA_COMMON_H

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#ifndef TRUE
#define TRUE 	1
#endif

#ifndef FALSE
#define FALSE 	0
#endif

#ifndef	NULL
#define NULL	0
#endif

#if (!defined(__cplusplus))
typedef	unsigned long			__BOOLEAN__;
#define bool					__BOOLEAN__
#endif

#define _GOODNESS(exp, r) if (!(exp)) return r

#define THREAD_INF_WAIT (-1)
#define THREAD_NO_WAIT (0)

#endif //REALTIMECAMERA_COMMON_H
