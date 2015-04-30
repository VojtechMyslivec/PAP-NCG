CC=nvcc
RM=rm -f
FLAGS=-arch=sm_20 -rdc=true -O3 -Xptxas -O3
LIBS=
DEFINE=
DEFINE += -D CACHE
DEFINE += -D MERENI

DIJKSTRA=./dijkstra
#FLOYD=./floyd-warshall

all: dijkstra 
# floyd

vypis: DEFINE += -D VYPIS
vypis: dijkstra
# floyd

#dijkstra: funkce cDijkstra dijkstraMain
$(DIJKSTRA): funkceSpolecne.o dijkstra.o cDijkstra.o mainDijkstra.o 
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

# floyd: funkce floydWarshall floydWarshallMain
# 	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(FLOYD) funkceSpolecne.o floydWarshall.o mainFloyd.o

#funkce:
funkceSpolecne.o: funkceSpolecne.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

#dijkstraMain:
mainDijkstra.o: main.cu
	$(CC) -c $(DEFINE) -D DIJKSTRA $(FLAGS) $(LIBS) -o $@ $^

#cDijkstra:
dijkstra.o: dijkstra.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

cDijkstra.o: cDijkstra.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

# floydWarshallMain:
# 	$(CC) -c $(DEFINE) -D FLOYDWARSHALL $(FLAGS) $(LIBS) -o mainFloyd.o main.cu
# 
# floydWarshall:
# 	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o floydWarshall.o floydWarshall.cu

clean:
	$(RM) funkceSpolecne.o dijkstra.o mainDijkstra.o cDijkstra.o $(DIJKSTRA) floydWarshall.o mainFloyd.o $(FLOYD)

