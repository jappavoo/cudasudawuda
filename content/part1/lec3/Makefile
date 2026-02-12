PYTHON:=python3.11
SRCS = $(wildcard *.drawio)
TARGETS = $(patsubst %.drawio,%.ipynb,$(SRCS))
IMGDIRS = $(patsubst %.drawio,%.imgs,$(SRCS))

.PHONY: all clean

%.ipynb: %.drawio
	./diox $< $*.imgs
	$(PYTHON) ./mknb.py $*.imgs -o $@

all: $(TARGETS)

clean:
	-rm -rf $(wildcard $(TARGETS) $(IMGDIRS))

