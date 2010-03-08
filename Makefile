CC = gcc

all: program

clean:
	-rm nnw

program:
	$(CC) parser.c nnw.c -o nnw -Wall --pedantic -lm
