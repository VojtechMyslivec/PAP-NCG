CC=g++
RM=rm -f
FLAGS=-Wall -pedantic
LIBS=
DEFINE=

DIJKSTRA=./dijkstra
FLOYD=./floyd-warshall

all: dijkstra floyd

dijkstra: funkce cDijkstra 
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o dijkstra.o dijkstra.cpp
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(DIJKSTRA) dijkstra.o funkceSpolecne.o cDijkstra.o 

floyd: funkce cFloydWarshall
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o floyd-warshall.o floyd-warshall.cpp 
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(FLOYD) floyd-warshall.o funkceSpolecne.o cFloydWarshall.o

funkce:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o funkceSpolecne.o funkceSpolecne.cpp

cDijkstra:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o cDijkstra.o cDijkstra.cpp

cFloydWarshall:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o cFloydWarshall.o cFloydWarshall.cpp

clean:
	$(RM) funkceSpolecne.o cDijkstra.o dijkstra.o $(DIJKSTRA) cFloydWarshall.o floyd-warshall.o $(FLOYD)

