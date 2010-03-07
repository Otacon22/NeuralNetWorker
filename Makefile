CC= gcc

all: program

clean:
	-rm nnw

program:
	$(CC) nnw.c -o nnw -Wall --pedantic -lm


