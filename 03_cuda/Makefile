CC      = nvcc
RM      = rm -f
FLAGS   = -gencode=arch=compute_30,code=compute_30 -rdc=true -O3 -Xptxas -O3 -Xptxas -v
LIBS    =
DEFINE  = # -D DEBUG
DEFINE += $(D)
DEFINE += -D CACHE
DEFINE += -D MERENI
DEFINE += -D SHARED

DIJKSTRA=dijkstra
FLOYD=floyd-warshall

all: $(DIJKSTRA) $(FLOYD)

rebuild: clean all

vypis: DEFINE += -D VYPIS
vypis: all

# alias
floyd: $(FLOYD)


$(DIJKSTRA): funkceSpolecne.o dijkstra.o cDijkstra.o mainDijkstra.o 
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

$(FLOYD): funkceSpolecne.o floydWarshall.o mainFloyd.o
	$(CC)    $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^


funkceSpolecne.o: funkceSpolecne.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^


mainDijkstra.o: main.cu
	$(CC) -c $(DEFINE) -D DIJKSTRA $(FLAGS) $(LIBS) -o $@ $^

dijkstra.o: dijkstra.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^

cDijkstra.o: cDijkstra.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^


mainFloyd.o: main.cu
	$(CC) -c $(DEFINE) -D FLOYDWARSHALL $(FLAGS) $(LIBS) -o $@ $^

floydWarshall.o: floydWarshall.cu
	$(CC) -c $(DEFINE) $(FLAGS) $(LIBS) -o $@ $^


clean: softclean
	$(RM) $(DIJKSTRA) $(FLOYD)

softclean:
	$(RM) funkceSpolecne.o dijkstra.o mainDijkstra.o cDijkstra.o floydWarshall.o mainFloyd.o

