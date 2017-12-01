MF=	Makefile

CC=	mpicc
CFLAGS= -O3

LFLAGS=	-lm

EXE=	imagempi \
        

SRC= \
	imagempi.c \
	arralloc.c \
	pgmio.c

INC=\
	arralloc.h  \
	pgmio.h

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .c .o

OBJ=	$(SRC:.c=.o)

.c.o:
	$(CC) $(CFLAGS) -std=c99 -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -std=c99 -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(INC)

$(OBJ):	$(MF)

clean:
	rm -f $(OBJ) $(EXE) core
