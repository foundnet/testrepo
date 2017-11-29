#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "arralloc.h"
#include "pgmio.h"


int main (int argc, char **argv)
{

  int i, j, iter, maxiter;
  char *filename;


  masterbuf = (double **) arralloc(sizeof(double), 2, M , N );
  buf       = (double **) arralloc(sizeof(double), 2, MP, NP);

  new  = (double **) arralloc(sizeof(double), 2, MP+2, NP+2);
  old  = (double **) arralloc(sizeof(double), 2, MP+2, NP+2);
  edge = (double **) arralloc(sizeof(double), 2, MP+2, NP+2);

  next = rank + 1;
  prev = rank - 1;
  
  
  int rank, size, next, prev;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /**
   *Virtual Topology Opetations: Create and manipulate the topology
  **/

  //Create the virtual Cartesian Topology
  MPI_Comm cart_comm;
  int dims[2] = {0,0};                  //This is a 2D decomposing.
  int period[2] = {1,0};                //The rows are peridic,the col is fixed.
  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  //Get the neighbers and self-node rank in virtual topology
  int left_nbr,right_nbr,top_nbr,bottom_nbr;
  MPI_Cart_shift( cart_comm, 0, 1, &left_nbr, &right_nbr );
  MPI_Cart_shift( cart_comm, 1, 1, &bottom_nbr, &top_nbr );
  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);
  



