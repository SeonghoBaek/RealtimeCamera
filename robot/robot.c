#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <termio.h>
#include <string.h>
#include <poll.h>
#include "bridge.h"
#import <stdio.h>

int uart_fd = -1;

void callback(void *cmd, int length)
{
    const char *str = "123\n";
    write(uart_fd, str, strlen(str)+1);

    return;
}

void looper()
{
    int fd[2];
    struct pollfd fds[1];

    pipe(fd);
    fds[0].fd = fd[0];
    fds[0].events = POLLIN;

    for (;;)
    {
        fds[0].revents = 0;
        poll(fds, 1, -1);
    }
}

int main(void)
{
    int fd;
    const char *str = "123\n";
    struct termios newtio;

    fd=open("/dev/ttyACM0", O_RDWR | O_NOCTTY );
    assert(fd != -1);
    

    // newtio <-- serial port setting.
    memset(&newtio, 0, sizeof(struct termios));
    newtio.c_cflag = B9600 | CS8 | CLOCAL | CREAD;
    newtio.c_iflag    = IGNPAR | ICRNL;
    newtio.c_oflag = 0;
    newtio.c_lflag = ~(ICANON | ECHO | ECHOE | ISIG);
    
    tcflush(fd, TCIFLUSH);
    tcsetattr(fd, TCSANOW, &newtio);


    uart_fd = fd;

    printf("%d\n", __LINE__);

    bridge_create("robot_bridge", callback);

    //write(fd, str, strlen(str)+1);

    looper();

    close(fd);

    return 0;
}
