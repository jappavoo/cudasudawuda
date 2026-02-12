PYTHON:=python3.11
SRCS = $(wildcard *.drawio)
TARGETS = $(patsubst %.drawio,%.ipynb,$(SRCS)) $(patsubst %.drawio,%.pdf,$(SRCS)) 
IMGDIRS = $(patsubst %.drawio,%.imgs,$(SRCS))

.PHONY: all clean

%.pdf: %.drawio
	DIOFORMAT=pdf ./diox $<

%.ipynb: %.drawio
	./diox $< $*.imgs
	$(PYTHON) ./mknb.py $*.imgs -o $@

all: $(TARGETS)

clean:
	-rm -rf $(wildcard $(TARGETS) $(IMGDIRS))

