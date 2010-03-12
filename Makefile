CC = gcc
FLAGS = -Wall --pedantic -O2 -march=native
LIBS = -lm

all: program

clean:
	-rm nnw
	-rm parser.o
	-rm nnw.o

program: parser.o nnw.o
	$(CC) parser.o nnw.o -o nnw $(LIBS)

parser.o:
	$(CC) -c parser.c -o parser.o $(FLAGS)

nnw.o:
	$(CC) -c nnw.c -o nnw.o $(FLAGS)
