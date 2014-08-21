CXX=g++
CXXFLAGS=-std=c++0x -O3 -fopenmp -lz -I. -DKENLM_MAX_ORDER=6
objs=lm/*.o util/*.o util/double-conversion/*.o

all: translator
translator: main.o translator.o lm.o ruletable.o vocab.o datastruct.o myutils.o $(objs)
	$(CXX) -o mebtg main.o translator.o lm.o ruletable.o vocab.o myutils.o datastruct.o $(objs) $(CXXFLAGS)

main.o: translator.h stdafx.h datastruct.h vocab.h ruletable.h lm.h myutils.h
translator.o: translator.h stdafx.h datastruct.h vocab.h ruletable.h lm.h myutils.h
lm.o: lm.h stdafx.h
ruletable.o: ruletable.h stdafx.h datastruct.h
vocab.o: vocab.h stdafx.h
datastruct.o: datastruct.h stdafx.h
myutils.o: myutils.h stdafx.h

clean:
	rm *.o
