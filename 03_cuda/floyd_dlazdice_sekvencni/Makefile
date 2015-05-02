CC      = g++
RM      = rm -f
FLAGS   = -Wall -pedantic -O3
LIBS    = 
DEFINE  =
DEFINE += 

FLOYD = floyd-warshall

all: $(FLOYD)

vypis: DEFINE += -D VYPIS
vypis: $(FLOYD)

$(FLOYD): funkceSpolecne.o floydWarshall.o mainFloyd.o
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

funkceSpolecne.o: funkceSpolecne.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

mainFloyd.o: main.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

floydWarshall.o: floydWarshall.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

clean:
	$(RM) funkceSpolecne.o floydWarshall.o mainFloyd.o $(FLOYD)

