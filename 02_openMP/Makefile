CC=g++
RM=rm
FLAGS=-Wall -pedantic
LIBS=-fopenmp
DEFINE=

DIJKSTRA=./dijkstra
FLOYD=./floyd-warshall

all: dijkstra floyd

dijkstra: funkce cDijkstra dijkstraMain
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(DIJKSTRA) dijkstra.o funkceSpolecne.o cDijkstra.o mainDijkstra.o

floyd: funkce floydWarshall floydWarshallMain
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(FLOYD) funkceSpolecne.o floydWarshall.o mainFloyd.o

funkce:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o funkceSpolecne.o funkceSpolecne.cpp

dijkstraMain:
	$(CC) -c $(DEFINE) -D DIJKSTRA $(FLAGS) $(LIBS) -o mainDijkstra.o main.cpp

cDijkstra:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o dijkstra.o dijkstra.cpp
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o cDijkstra.o cDijkstra.cpp

floydWarshallMain:
	$(CC) -c $(DEFINE) -D FLOYDWARSHALL $(FLAGS) $(LIBS) -o mainFloyd.o main.cpp

floydWarshall:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o floydWarshall.o floydWarshall.cpp

clean:
	$(RM) funkceSpolecne.o dijkstra.o mainDijkstra.o cDijkstra.o $(DIJKSTRA) floydWarshall.o mainFloyd.o $(FLOYD)

