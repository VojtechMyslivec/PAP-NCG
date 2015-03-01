CC=g++
RM=rm -f
FLAGS=-Wall -pedantic
LIBS=

OUTPUT=./dijkstra.out

all: 
	$(CC) $(FLAGS) $(LIBS) dijkstra.cpp -o $(OUTPUT)

debug: 
	$(CC) -D DEBUG $(FLAGS) $(LIBS) dijkstra.cpp -o $(OUTPUT)

debug2: 
	$(CC) -D DEBUG2 $(FLAGS) $(LIBS) dijkstra.cpp -o $(OUTPUT)

valgrind: 
	$(CC) -g $(FLAGS) $(LIBS) dijkstra.cpp -o $(OUTPUT)

clean:
	$(RM) $(OUTPUT)
