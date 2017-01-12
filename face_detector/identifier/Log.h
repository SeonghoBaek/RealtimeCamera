//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_LOG_H
#define REALTIMECAMERA_LOG_H

#define LOG_PRINT_ENABLE 1
#define LOG_ERROR -1
#define LOG_WARN 0
#define LOG_INFO 1
#define LOG_DEBUG 2
#define LOG_VERBOSE 3
#define RELEASE_MODE 0

#if (RELEASE_MODE == 1)
#define DEBUG_MODE 0
#else
#define DEBUG_MODE 1
#endif

#define  LOGE(...) \
           LOG_PrintMessage(LOG_ERROR,__func__,__LINE__,__VA_ARGS__)

#define  LOGW(...) \
           LOG_PrintMessage(LOG_WARN,__func__,__LINE__,__VA_ARGS__)


#define  LOGI(...) \
           LOG_PrintMessage(LOG_INFO,__func__,__LINE__,__VA_ARGS__)

#if (DEBUG_MODE == 1)
#define  LOGD(...) \
           LOG_PrintMessage(LOG_DEBUG,__func__,__LINE__,__VA_ARGS__)
#else
#define LOGD(...)
#endif

#define  LOGV(...) \
           LOG_PrintMessage(LOG_VERBOSE,__func__,__LINE__,__VA_ARGS__)

void LOG_PrintMessage(int type, const char *func, int line, const char *fmt, ...);
#endif //REALTIMECAMERA_LOG_H
