CC=g++
RM=rm -f
FLAGS=-Wall -pedantic -O3
LIBS=-fopenmp
DEFINE=
DEFINE += -D MERENI

DIJKSTRA=dijkstra
FLOYD=floyd-warshall

all: $(DIJKSTRA) $(FLOYD)

# alias
floyd: $(FLOYD)

vypis: DEFINE += -D VYPIS
vypis: $(DIJKSTRA) $(FLOYD)

$(DIJKSTRA): funkceSpolecne.o dijkstra.o cDijkstra.o mainDijkstra.o
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

$(FLOYD): funkceSpolecne.o floydWarshall.o mainFloyd.o
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

funkceSpolecne.o: funkceSpolecne.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

mainDijkstra.o: main.cpp
	$(CC) -c $(DEFINE) -D DIJKSTRA $(FLAGS) $(LIBS) -o $@ $^

cDijkstra.o: cDijkstra.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

dijkstra.o: dijkstra.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

mainFloyd.o: main.cpp
	$(CC) -c $(DEFINE) -D FLOYDWARSHALL $(FLAGS) $(LIBS) -o $@ $^

floydWarshall.o: floydWarshall.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

clean:
	$(RM) funkceSpolecne.o dijkstra.o mainDijkstra.o cDijkstra.o $(DIJKSTRA) floydWarshall.o mainFloyd.o $(FLOYD)

