CC=g++
RM=rm -f
FLAGS=-Wall -pedantic
LIBS=-fopenmp
DEFINE=

DIJKSTRA=./dijkstra
FLOYD=./floyd-warshall

all: dijkstra floyd

dijkstra: funkce cDijkstra 
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o dijkstra.o dijkstra.cpp
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(DIJKSTRA) dijkstra.o funkceSpolecne.o cDijkstra.o 

floyd: funkce floydWarshall
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o floyd.o floyd.cpp 
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(FLOYD) floyd.o funkceSpolecne.o floydWarshall.o

funkce:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o funkceSpolecne.o funkceSpolecne.cpp

cDijkstra:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o cDijkstra.o cDijkstra.cpp

floydWarshall:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o floydWarshall.o floydWarshall.cpp

clean:
	$(RM) funkceSpolecne.o cDijkstra.o dijkstra.o $(DIJKSTRA) floydWarshall.o floyd.o $(FLOYD)

