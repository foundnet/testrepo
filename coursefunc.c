#include "coursefunc.h"

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
        //MPI_Issend(&cur_image[1][range[1]],1,chalo,nbr_rank[r],10,*pcomm,&send_req[r]);
        //MPI_Irecv(&cur_image[1][range[1]+1],1,chalo,nbr_rank[r],10,*pcomm,&recv_req[r]);
        break;
      case BOTTOM:
        //MPI_Issend(&cur_image[1][1],1,chalo,nbr_rank[r],10,*pcomm,&send_req[r]);
        //MPI_Irecv(&cur_image[1][0],1,chalo,nbr_rank[r],10,*pcomm,&recv_req[r]);
        break;
    }
  }
}

void Iwaithalos(int*nbr_rank, MPI_Request send_req[], MPI_Request recv_req[]) {
  MPI_Status status;
  for (int r=0 ; r < 4 ; r++)  {
    if (nbr_rank[r] == MPI_PROC_NULL)  continue;
    MPI_Wait(&send_req[r], &status);
    MPI_Wait(&send_req[r], &status);
  }
}

double calculateimg(double **new_img, double **old_img, double **edge, int starti,   \
                  int startj, int endi, int endj, double *sumnew) {
  double delta_max = 0;
  for (int i=starti ; i <= endi; i++ )  {
    for (int j=startj ; j <= endj; j++)  {
      new_img[i][j]=0.25*(old_img[i-1][j]+old_img[i+1][j]+old_img[i][j-1]+old_img[i][j+1]-edge[i-1][j-1]);
      *sumnew = *sumnew + new_img[i][j];
      double delta = fabs(new_img[i][j]-old_img[i][j]);
      if (delta > delta_max)    delta_max = delta;
    }
  }
  return delta_max; 
}