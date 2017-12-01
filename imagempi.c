#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>
#include <string.h>

#include "arralloc.h"
#include "pgmio.h"


double boundaryval(int i, int m);
typedef enum _nbr {LEFT,RIGHT,TOP,BOTTOM} nbr;

// The No.0 node scatter edge data to every node. The other nodes receive edge data.
double ** scatter_vector(double **sendbuf, int *block_size, int N_modi,int cur_rank, int comm_size, MPI_Datatype DATATYPE, MPI_Comm *pcomm) {
  double **edge = (double **) arralloc(sizeof(double), 2, block_size[0], N_modi);
  MPI_Request *request = (MPI_Request *)malloc( comm_size*sizeof(MPI_Request) );
  MPI_Status status;

  if (cur_rank == 0)  {
    int cur_coods[2] = {0,0};
    // The No.0 node send data to the other nodes.
    for (int i=1; i < comm_size; i++) {
      MPI_Cart_coords(*pcomm, i, 2, cur_coods) ;
      MPI_Issend(&sendbuf[cur_coods[0]*block_size[0]][cur_coods[1]*block_size[1]],1,DATATYPE,i,5,*pcomm,&request[i]);
      printf("CART_RANK:0  SCATTER SEND-%d POS-I-%d J-%d \n",i,cur_coods[0]*block_size[0],cur_coods[1]*block_size[1]);
    }
    // The No.0 node copy data to its own edge data vector.
    for (int i=0; i<block_size[0]; i++)
      for (int j=0; j<block_size[1]; j++)
        edge[i][j] = sendbuf[i][j];
    // The No.0 node wait for all the responses.
    for (int i=1 ; i < comm_size ; i++) {
      MPI_Wait(&request[i], &status);
    }
    printf("CART_RANK:0 SCATTER SEND ALL DONE\n");
  }
  else {
    // The other nodes receive data and put them in edge vector.
    MPI_Recv(&edge[0][0], 1, DATATYPE, 0, 5, *pcomm, &status);
    printf("CART_RANK:%d  SCATTER RECVED\n",cur_rank);
  }

  free(request);
  return edge;
}

int gather_vector(double **recvbuf,double **localimg,int *block_size,int cur_rank,int comm_size,MPI_Datatype DATATYPE,MPI_Comm *pcomm) {
  MPI_Request *request = (MPI_Request *)malloc( comm_size*sizeof(MPI_Request) );
  MPI_Status status;
  if (cur_rank == 0)  {
    int rcv_coods[2] = {0,0};
    // The No.0 node recv data from the other nodes and save it in a temporary buffer.
    for (int i=1; i < comm_size; i++) {
      MPI_Cart_coords(*pcomm, i, 2, rcv_coods) ;
      MPI_Irecv(&recvbuf[rcv_coods[0]*block_size[0]][rcv_coods[1]*block_size[1]],1,DATATYPE,i,6,*pcomm,&request[i]);
      printf("CART_RANK:0  GATHER RECV-%d POS-I-%d J-%d \n",i,rcv_coods[0]*block_size[0],rcv_coods[1]*block_size[1]);
    }
    // The No.0 node copy local edge data to the complete data vector.
    for (int i=0; i<block_size[0]; i++)
      for (int j=0; j<block_size[1]; j++)
        recvbuf[i][j] = localimg[i][j] ;
    // The No.0 node wait for all the receives finished.
    for (int i=1 ; i < comm_size ; i++) {
      MPI_Wait(&request[i], &status);
    }
    printf("CART_RANK:0 GATHER RECV ALL DONE\n");
 }
  else {
    // The other nodes send their local image data to No.0 node.
    MPI_Ssend(&localimg[0][0], 1, DATATYPE, 0, 6, *pcomm);
    printf("CART_RANK:%d  GATHER SEND\n",cur_rank);
  }

  free(request);
  return 1;
}


void Iswaphalos(double **cur_image, int*nbr_rank, int *range, MPI_Datatype rhalo,  \
               MPI_Datatype chalo, MPI_Comm *pcomm, MPI_Request *send_req, MPI_Request *recv_req) {
  for (int r=0 ; r < 4 ; r++)  {
    if (nbr_rank[r] == MPI_PROC_NULL)  continue;
    switch (r) {
      case LEFT:
        MPI_Issend(&cur_image[1][1],1,rhalo,nbr_rank[r],10,*pcomm,&send_req[r]);
        MPI_Irecv (&cur_image[0][1],1,rhalo,nbr_rank[r],10,*pcomm,&recv_req[r]);
        break;
      case RIGHT:
        MPI_Issend(&cur_image[range[0]][1],1,rhalo,nbr_rank[r],10,*pcomm,&send_req[r]);
        MPI_Irecv(&cur_image[range[0]+1][1],1,rhalo,nbr_rank[r],10,*pcomm,&recv_req[r]);
        break;
      case TOP:
        MPI_Issend(&cur_image[1][range[1]],1,chalo,nbr_rank[r],10,*pcomm,&send_req[r]);
        MPI_Irecv(&cur_image[1][range[1]+1],1,chalo,nbr_rank[r],10,*pcomm,&recv_req[r]);
        break;
      case BOTTOM:
        MPI_Issend(&cur_image[1][1],1,chalo,nbr_rank[r],10,*pcomm,&send_req[r]);
        MPI_Irecv(&cur_image[1][0],1,chalo,nbr_rank[r],10,*pcomm,&recv_req[r]);
    }
  }
}

void Iwaithalos(int*nbr_rank, MPI_Request *send_req, MPI_Request *recv_req) {
  MPI_Status status;
  for (int r=0 ; r < 4 ; r++)  {
    if (nbr_rank[r] == MPI_PROC_NULL)  continue;
    MPI_Wait(&send_req[r], &status);
    MPI_Wait(&send_req[r], &status);
  }
}

void calculateimg(double **new_img, double **old_img, double **edge, int starti, int startj, int endi, int endj) {
  for (int i=starti ; i <= endi; i++ )  {
    for (int j=startj ; j <= endj; j++)  {
      new_img[i][j]=0.25*(old_img[i-1][j]+old_img[i+1][j]+old_img[i][j-1]+old_img[i][j+1]-edge[i-1][j-1]);
    }
  } 
}

int main (int argc, char **argv) {
  double **edge, **masterbuf, **sendbuf, **buf;
  double **temp = (double **)arralloc(sizeof(double), 2, 1, 1 );
  sendbuf = temp;
  masterbuf = temp;

  int i, j, iter, maxiter;
  //, N, M, M_modi, N_modi;
  int img_size[2] = {0,0};
  int img_modisize[2] = {0,0};
  int block_size[2] = {0,0};
  int nbr_rank[4] = {-1,-1,-1,-1};

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
  MPI_Cart_shift( cart_comm, 0, 1, &nbr_rank[LEFT], &nbr_rank[RIGHT] );
  MPI_Cart_shift( cart_comm, 1, 1, &nbr_rank[BOTTOM], &nbr_rank[TOP] );
  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);


  printf("CART_RANK:%d  O_RANK:%d DIM[0]%d DIM[1]%d Left:%d Right:%d Top:%d Bottom:%d\n", \
         cart_rank,rank,dims[0],dims[1],nbr_rank[LEFT],nbr_rank[RIGHT],nbr_rank[TOP],nbr_rank[BOTTOM]);

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
    pgmsize(filename,&img_size[0], &img_size[1]);
    block_size[0] = img_size[0]%dims[0] == 0? img_size[0]/dims[0]:img_size[0]/dims[0]+1;
    block_size[1] = img_size[1]%dims[1] == 0? img_size[1]/dims[1]:img_size[1]/dims[1]+1;
    //Enlarge the data vector's size to make it can be divided exactly by the dims
    //So, we can use derived datatype to distribute and gather data
    img_modisize[0] = block_size[0] * dims[0];
    img_modisize[1] = block_size[1] * dims[1];

    masterbuf = (double **)arralloc(sizeof(double), 2, img_size[0] , img_size[1] );
    pgmread(filename, &masterbuf[0][0], img_size[0], img_size[1]);
    
    if (img_size[0] == img_modisize[0] && img_size[1] == img_modisize[1])  sendbuf = masterbuf;
    else {
      sendbuf = (double **)arralloc(sizeof(double), 2, img_modisize[0] , img_modisize[1] );
      for (int i=0; i < img_modisize[0] ; i++)
        for (int j=0; j < img_modisize[1] ; j++) {
            if ( i<img_size[0] && j<img_size[1] )  sendbuf[i][j] = masterbuf[i][j] ;
            else sendbuf[i][j] = DBL_MAX;
        }
    }
  }

  MPI_Bcast(&block_size, 2, MPI_INT, 0, cart_comm) ;
  MPI_Bcast(&img_size, 2, MPI_INT, 0, cart_comm) ;
  MPI_Bcast(&img_modisize, 2, MPI_INT, 0, cart_comm) ;
  printf("CART_RANK:%d  After Bcast BLOCK:%d %d MN:%d %d MODI:%d %d\n",cart_rank,block_size[0],block_size[1],img_size[0],img_size[1],img_modisize[0],img_modisize[1]);
 
  // Create new derived datatype to transfer data
  MPI_Datatype DT_BLOCK;
  MPI_Type_vector(block_size[0], block_size[1], img_modisize[1], MPI_DOUBLE, &DT_BLOCK);
  MPI_Type_commit(&DT_BLOCK);
  
  edge = (double**)scatter_vector(sendbuf, block_size, img_modisize[1], cart_rank, size, DT_BLOCK, &cart_comm);

  // Set the working range by iterate the edge vector.
  int range[2] = {block_size[0] , block_size[1]};
  for (i=0 ; i < block_size[0] ; i++)  {
    if (edge[i][0] == DBL_MAX) {
      range[0] = i;
      break ;
    }
  }
  for (j=0 ; j < block_size[1] ; j++)  {
    if (edge[0][j] == DBL_MAX) {
      range[1] = j;
      break ;
    }
  }
  printf("CART_RANK:%d  RANGE BLOCK:R-SIZE[0] %d R-SIZE[1] %d \n",cart_rank,range[0],range[1]);

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
  double val;
  MPI_Cart_coords(cart_comm, cart_rank, 2, cur_coods) ;
  if (nbr_rank[BOTTOM] == MPI_PROC_NULL )  {
    for ( i=1 ; i < range[0]+1 ; i++)  {
      /* compute sawtooth value */
      val = boundaryval(i+cur_coods[0]*block_size[0], img_size[0]);

      odd[i][0] = (int)(255.0*val);
      even[i][0] = (int)(255.0*val);
    }
    printf("CART_RANK:%d  SETFIXB BOT [FROM%d TO %d][0] M-%d\n",cart_rank,1+cur_coods[0]*block_size[0],range[0]+cur_coods[0]*block_size[0],img_size[0]);
  }
  if (nbr_rank[TOP] == MPI_PROC_NULL )  {
    for ( i=1 ; i < range[0]+1 ; i++)  {
      /* compute sawtooth value */
      val = boundaryval(i+cur_coods[0]*block_size[0], img_size[0]);

      odd[i][range[1]+1] = (int)(255.0*(1.0-val));
      even[i][range[1]+1] = (int)(255.0*(1.0-val));
    }
    printf("CART_RANK:%d  SETFIXB TOP [FROM%d TO %d][%d] M-%d\n",cart_rank,1+cur_coods[0]*block_size[0],range[0]+cur_coods[0]*block_size[0],range[1]+1,img_size[0]);
  }

  // Create the derived data type to switch the halo row. 
  MPI_Datatype DT_ROWHALO;
  MPI_Type_vector(1, block_size[1], block_size[1], MPI_DOUBLE, &DT_ROWHALO);
  MPI_Type_commit(&DT_ROWHALO);

  // Create the derived data type to switch the halo column
  MPI_Datatype DT_COLHALO;
  MPI_Type_vector(block_size[0], 1, block_size[1]+2, MPI_DOUBLE, &DT_COLHALO);
  MPI_Type_commit(&DT_COLHALO);  
  
  // The calculation and the halo switch begins
  iter = 0 ;
  double delta_max = 1;
  pold = odd;
  pnew = even;
  MPI_Request send_req[4], recv_req[4];

/*
  while (iter < 500) {
    // First, swap the halos using the old map
    Iswaphalos(pold, nbr_rank, range, DT_ROWHALO, DT_COLHALO, &cart_comm, send_req, recv_req);
    // Second, calculate the centre cells which will not be affected by halos
    calculateimg(pnew, pold, edge, 2, 2, range[0]-1, range[1]-1);
    //Third, wait for all the asyn tasks finished.
    Iwaithalos(nbr_rank, send_req, recv_req);
    //Finally, calculate the cells that will be affected by halos
    calculateimg(pnew, pold, edge, 1, 1, 1, range[1]);
    calculateimg(pnew, pold, edge, range[0], 1, range[0], range[1]);
    calculateimg(pnew, pold, edge, 2, 1, range[0]-1, 1);
    calculateimg(pnew, pold, edge, 2, range[1], range[0]-1, range[1]);
    //Swap the pold and pnew pointer 
    double **pswap = pnew;
    pold = pnew;
    pnew = pswap;
    iter ++;
  }  

  // After the calculation , the No.0 node gather the data together and save to file.
  for (i=1 ; i < range[0]+1 ;i++)
     for (j=1 ; j < range[1]+1 ; j++)
       edge[i-1][j-1] = pold[i][j] ;
*/
//  memset(&masterbuf[0][0], 0, sizeof(double)*img_size[0]*img_size[1]); 
  gather_vector(sendbuf, edge, block_size, cart_rank, size, DT_BLOCK, &cart_comm);
  
  if (cart_rank == 0)  {
    if (sendbuf != masterbuf) 
      for (i=0 ; i < img_size[0] ; i++)
        for (j=0 ; j < img_size[1] ; j++)
          masterbuf[i][j] = sendbuf[i][j];
    
    pgmwrite("parallelimg.pgm", &masterbuf[0][0], img_size[0], img_size[1]);
  }

  printf("I didn't deadlock, yeah!\n");

  MPI_Finalize();
}


double boundaryval(int i, int m)
{
  double val;

  val = 2.0*((double)(i-1))/((double)(m-1));
  if (i >= m/2+1) val = 2.0-val;
  
  return val;
}