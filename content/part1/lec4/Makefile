PYTHON:=python3.11
SRCS = $(wildcard *.drawio)
TARGETS = $(patsubst %.drawio,%.pdf,$(SRCS)) 
IMGDIRS = $(patsubst %.drawio,%.imgs,$(SRCS))

.PHONY: all clean

%.pdf: %.drawio
	./diox $< $*.imgs
	DIOFORMAT=pdf ./diox $<

%.ipynb: %.drawio
	./diox $< $*.imgs
	$(PYTHON) ./mknb.py $*.imgs -o $@

all: $(TARGETS)

clean:
	-rm -rf $(wildcard $(TARGETS) $(IMGDIRS))

