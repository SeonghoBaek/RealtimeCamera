SUBDIRS = sparse identifier extractor

.PHONY : all target $(SUBDIRS) clean

all : $(SUBDIRS) ;

$(SUBDIRS) :
	make -j4 -C $@

target:
	@for dir in $(SUBDIRS); do \
		make -j4 -C $$dir $@ ;\
	done

clean:
	@for dir in $(SUBDIRS); do \
		make -C $$dir $@ ;\
	done
