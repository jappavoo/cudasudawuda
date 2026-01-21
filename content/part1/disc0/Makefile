CC = gcc
SHELL := /bin/bash -O extglob
CFLAGS += -O0 -g
LDFLAGS += -lpthread

.PHONY: clean

create0: create0.c
	${CC} $(CFLAGS) $(LDFLAGS) $< -o $@ 2>&1 | tee make.out

clean:
	-@${RM} !(Makefile|siggen.c|README.md)
