#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>

#include "arralloc.h"
#include "pgmio.h"


double boundaryval(int i, int m);

// The No.0 node scatter edge data to every node. The other nodes receive edge data.
double ** scatter_vector(double **sendbuf, int*block_size, int N_modi,int cur_rank, int comm_size, MPI_Datatype *pDATATYPE, MPI_Comm *pcomm) {
  double **edge = (double **) arralloc(sizeof(double), 2, block_size[0], N_modi);
  if (cur_rank == 0)  {
    MPI_Request *request = (MPI_Request *)malloc( comm_size*sizeof(MPI_Request) );
    MPI_Status status;
    int cur_coods[2] = {0,0};
    // The No.0 node send data to the other nodes.
    for (int i=1; i < comm_size; i++) {
      MPI_Cart_coords(*pcomm, i, 2, cur_coods) ;
      MPI_Issend(&sendbuf[cur_coods[0]*block_size[0]][cur_coods[1]*block_size[1]],1,*pDATATYPE,i,5,*pcomm,&request[i-1]);
    }
    // The No.0 node copy data to its own edge data vector.
    for (i=0; i<block_size[0]; i++)
      for (j=0; j<block_size[1]; j++)
        edge[i][j] = sendbuf[i][j];
    // The No.0 node wait for all the responses.
    for (i=0 ; i < comm_size ; i++) {
      MPI_Wait(&request[i], &status);
    }
  }
  else {
    // The other nodes receive data and put them in edge vector.
    MPI_Recv(&edge[0][0], 1, *pDATATYPE, 0, 5, *pcomm, &status);
  }

  free(request);
  return edge;
}

int gather_vector(double **recvbuf,double **localimg,int*block_size,int cur_rank,int comm_size,MPI_Dataype *pDATATYPE,MPI_Comm *pcomm) {
  if (cur_rank == 0)  {
    MPI_Request *request = (MPI_Request *)malloc( comm_size*sizeof(MPI_Request) );
    MPI_Status status;
    int rcv_coods[2] = {0,0};
    // The No.0 node recv data from the other nodes and save it in a temporary buffer.
    for (int i=1; i < comm_size; i++) {
      MPI_Cart_coords(*pcomm, i, 2, rcv_coods) ;
      MPI_Irecv(&recvbuf[rcv_coods[0]*block_size[0]][rcv_coods[1]*block_size[1]],1,*pDATATYPE,i,6,*pcomm,&request[i-1]);
    }
    // The No.0 node copy local edge data to the complete data vector.
    for (int i=0; i<block_size[0]; i++)
      for (j=0; j<block_size[1]; j++)
        recvbuf[i][j] = localimg[i][j] ;
    // The No.0 node wait for all the receives finished.
    for (int i=0 ; i < comm_size ; i++) {
      MPI_Wait(&request[i], &status);
    }
  }
  else {
    // The other nodes send their local image data to No.0 node.
    MPI_Ssend(&localimg[0][0], 1, *pDATATYPE, 0, 6, *pcomm,&status);
  }

  free(request);
  return 1;
}


int main (int argc, char **argv)
{
  double **edge, **masterbuf, **sendbuf, **buf;
  double temp[1][1] = {1};
  sendbuf = &temp;
  masterbuf = &temp;

  int i, j, iter, maxiter, N, M, M_modi, N_modi;
  int block_size[2] = {0,0};

  char *filename;

  int rank, size;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/**
*   STEP 1 : Virtual Topology Opetations
*            Create and manipulate the topology for communication
**/

  //Create the virtual Cartesian Topology
  MPI_Comm cart_comm;
  int dims[2] = {0,0};                  //This is a 2D decomposing.
  int period[2] = {1,0};                //The rows are peridic,the col is fixed.
  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &cart_comm);

  //Get the neighbers and self-node rank in virtual topology
  int left_nbr,right_nbr,top_nbr,bottom_nbr;
  MPI_Cart_shift( cart_comm, 0, 1, &left_nbr, &right_nbr );
  MPI_Cart_shift( cart_comm, 1, 1, &bottom_nbr, &top_nbr );
  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);

/**
*   STEP 2 : 
*            
**/

  //The No.0 node read the edge data and calculate the subvector's size.
  //Then it broadcast the size data and scatter the edge data to the other nodes.
  if(cart_rank == 0)  {
    if (argc > 1) {
      filename = argv[1];
    } 
    else {
      printf("Usage: ./imagecalc <edge file name>\n");
      return 0;
    }
    pgmsize(filename,&M, &N);
    block_size[0] = M%dims[0] == 0? M/dims[0]:M/dims[0]+1;
    block_size[1] = N%dims[1] == 0? N/dims[1]:N/dims[1]+1;
    //Enlarge the data vector's size to make it can be divided exactly by the dims
    //So, we can use derived datatype to distribute and gather data
    M_modi = block_size[0] * dims[0];
    N_modi = block_size[1] * dims[1];

    masterbuf = (double **)arralloc(sizeof(double), 2, M , N );
    pgmread(filename, &masterbuf[0][0], M, N);
    
    if (M_modi == M && N_modi == N)  sendbuf = masterbuf;
    else {
      sendbuf = (double **)arralloc(sizeof(double), 2, M_modi , N_modi );
      for (int i=0; i<M_modi; i++)
        for (int j=0; j<N_modi; j++) {
            if ( i<M && j<N )  sendbuf[i][j] = masterbuf[i][j] ;
            else sendbuf[i][j] = DBL_MAX;
        }
    }
/*  printf("Processing %d x %d image on %d processes\n", M, N, P);
    printf("Number of iterations = %d\n", MAXITER);
  
    filename = "edge192x128.pgm";
  
      printf("\nReading <%s>\n", filename);
     printf("\n");
*/
  }

  MPI_Bcast(&block_size, 2, MPI_INT, 0, cart_comm) ;
  MPI_Bcast(&N_modi, 1, MPI_INT, 0, cart_comm) ;

  // Create new derived datatype to transfer data
  MPI_Type DT_BLOCK;
  MPI_Type_vector(block_size[0], block_size[1], N_modi, MPI_DOUBLE, &DT_BLOCK);
  MPI_Type_commit(&DT_BLOCK);
  
  edge = (double**)scatter_vector(sendbuf, &block_size, cart_rank, size, &DT_BLOCK, &cart_comm);

  double **pnew, **pold;
  double **odd  = (double **) arralloc(sizeof(double), 2, block_size[0]+2, block_size[1]+2);
  double **even  = (double **) arralloc(sizeof(double), 2, block_size[0]+2, block_size[1]+2);
  
  // Initialize the image vector.
  for ( i=0; i < block_size[0]+2 ; i++ ) {
    for ( j=0 ; j < block_size[1]+2 ; j++)  {
	    odd[i][j] = 255.0;
      even[i][j] = 255.0;
    }
  }

  // Set the fixed boundary value.
  int cur_coods[2] = {0,0};
  MPI_Cart_coords(cart_comm, cart_rank, 2, cur_coods) ;
  if (bottom_nbr == MPI_PROC_NULL )  {
    for ( i=1 ; i < block_size[0]+1 ; i++)  {
      /* compute sawtooth value */
      val = boundaryval(i+cur_coods[0]*block_size[0], M);

      odd[i][0] = (int)(255.0*val);
      even[i][0] = (int)(255.0*val);
    }
  }
  else if (top_nbr == MPI_PROC_NULL )  {
    for ( i=1 ; i < block_size[0]+1 ; i++)  {
      /* compute sawtooth value */
      val = boundaryval(i+cur_coods[0]*block_size[0], M);

      odd[i][block_size[1]+1] = (int)(255.0*(1.0-val));
      even[i][block_size[1]+1] = (int)(255.0*(1.0-val));
    }
  }
  iter = 0 ;
}


double boundaryval(int i, int m)
{
  double val;

  val = 2.0*((double)(i-1))/((double)(m-1));
  if (i >= m/2+1) val = 2.0-val;
  
  return val;
}