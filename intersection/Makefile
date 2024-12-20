CC = g++ -std=c++11 -fopenmp

CFLAGS = -c -Wall -O3

all: interx

interx: main.o intersection.o Tools.o
	$(CC) main.o intersection.o Tools.o -o interx

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

intersection.o: intersection.cpp
	$(CC) $(CFLAGS) intersection.cpp

Tools.o: Tools.cpp
	$(CC) $(CFLAGS) Tools.cpp

clean:
	rm -rf *o interx
