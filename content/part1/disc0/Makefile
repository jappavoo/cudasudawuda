CC = gcc
SHELL := /bin/bash -O extglob
CFLAGS += -O0 -g
LDFLAGS += -lpthread

.PHONY: clean

all: create0 create1 create2 create3 create3.5 create4 args0 args1 barrier0 counter0 counter1

%: %.c
	${CC} $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	-@${RM} !(Makefile|*.c|README.md)
