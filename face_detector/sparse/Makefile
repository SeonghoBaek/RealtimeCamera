LDLFAGS =
LDFLAGS += `pkg-config --libs opencv`

include_dirs := .
src_dir := .

SRC =
SRC += utils.cpp test.cpp train.cpp

SRCS += $(addprefix $(src_dir)/,$(SRC))

OBJS = $(subst .cpp,.o,$(SRCS))

CPPFLAGS += $(addprefix -I,$(include_dirs))

CPPFLAGS += `pkg-config --cflags opencv`

LDFLAGS += -lpthread

all : $(OBJS)
	g++ -shared -g -ggdb -o ../lib/libsparse.so $(OBJS) $(LDFLAGS)
	rm -f $(OBJS)

%.o : %.cpp
	g++ -fpic -c $< $(CPPFLAGS) -g -ggdb -o $(@)

clean :
	rm -f $(OBJS)
	rm -f ../lib/libsparse.so