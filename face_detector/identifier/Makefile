include_dirs := . /usr/local/include/hiredis ../sparse
src_dir := .

SRC = 
SRC += Classifier.cpp Lock.cpp VectorClassFinder.cpp VectorSubscriber.cpp Log.cpp
SRC += VectorNetSubscriber.cpp Identifier.cpp Looper.cpp VectorQueue.cpp
SRC += RedisIO.cpp

SRCS += $(addprefix $(src_dir)/,$(SRC))

OBJS = $(subst .cpp,.o,$(SRCS))

CPPFLAGS += $(addprefix -I,$(include_dirs))

LDFLAGS += -levent -lhiredis -lpthread

LDFLAGS += -L../lib -lsparse

all : $(OBJS)
	g++ -shared -g -ggdb -o ../lib/libidentifier.so $(OBJS) $(LDFLAGS)
	rm -f $(OBJS)	

%.o : %.cpp
	g++ -fpic -c $< $(CPPFLAGS) -g -ggdb -o $(@)

clean :
	rm -f $(OBJS)
	rm -f ../lib/libidentifier.so
