include_dirs := . /usr/local/include/hiredis
src_dir := .

SRC = 
SRC += Classifier.cpp Lock.cpp VectorClassFinder.cpp VectorSubscriber.cpp Log.cpp
SRC += VectorNetSubscriber.cpp Identifier.cpp Looper.cpp VectorQueue.cpp
SRC += RedisIO.cpp

SRCS += $(addprefix $(src_dir)/,$(SRC))

OBJS = $(subst .cpp,.o,$(SRCS))

CPPFLAGS += $(addprefix -I,$(include_dirs))

LDFLAGS += -levent -lhiredis -lpthread

all : $(OBJS)
	g++ -shared -o ../lib/libidentifier.so $(OBJS) $(LDFLAGS)
	rm -f $(OBJS)	

%.o : %.cpp
	g++ -fpic -c $< $(CPPFLAGS) -o $(@)

clean :
	rm -f $(OBJS)
	rm -f ../lib/libidentifier.so
