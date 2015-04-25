CC=nvcc
RM=rm -f
FLAGS=-g -rdc=true -arch=sm_20 # -O3
LIBS=
DEFINE=

DIJKSTRA=./dijkstra
#FLOYD=./floyd-warshall

all: dijkstra 
	# floyd

dijkstra: funkce cDijkstra dijkstraMain
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -lcudadevrt -o $(DIJKSTRA) dijkstra.o funkceSpolecne.o cDijkstra.o mainDijkstra.o

# floyd: funkce floydWarshall floydWarshallMain
# 	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $(FLOYD) funkceSpolecne.o floydWarshall.o mainFloyd.o

funkce:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o funkceSpolecne.o funkceSpolecne.cu

dijkstraMain:
	$(CC) -c $(DEFINE) -D DIJKSTRA $(FLAGS) $(LIBS) -o mainDijkstra.o main.cu

cDijkstra:
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o dijkstra.o dijkstra.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o cDijkstra.o cDijkstra.cu

# floydWarshallMain:
# 	$(CC) -c $(DEFINE) -D FLOYDWARSHALL $(FLAGS) $(LIBS) -o mainFloyd.o main.cu
# 
# floydWarshall:
# 	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o floydWarshall.o floydWarshall.cu

clean:
	$(RM) funkceSpolecne.o dijkstra.o mainDijkstra.o cDijkstra.o $(DIJKSTRA) floydWarshall.o mainFloyd.o $(FLOYD)

