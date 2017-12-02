#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <float.h>
#include <string.h>

#include "arralloc.h"

typedef enum _nbr {LEFT,RIGHT,TOP,BOTTOM} nbr;

/**
 * @brief distribute datas in a commu group, the rank 0 send data ,others receive.
 * @param sendbuf       the data should be sent by rank 0
 * @param block_size    the size of a block 
 * @param N_modi        the modified j value of image
 * @param cur_rank      current rank number
 * @param comm_size     the size of a communitor
 * @param DATATYPE      the customized datatype  
 * @return double **    new received new edge data vector
**/
double ** scatter_vector(double **sendbuf, int *block_size, int N_modi,int cur_rank, int comm_size, MPI_Datatype DATATYPE, MPI_Comm *pcomm);

/**
 * @brief Gather datas from a commu group, the rank 0 recv data ,others send.
 * @param recvbuf      the data should be stored by rank 0
 * @param localimg     the local image vector that will send 
 * @param block_size    the size of a block
 * @param N_modi        the modified j value of image
 * @param cur_rank      current rank number
  * @param comm_size     the size of a communitor
 * @param DATATYPE      the customized datatype  * @return double **    new received new edge data vector
**/
int gather_vector(double **recvbuf,double **localimg,int *block_size,int cur_rank,int comm_size,MPI_Datatype DATATYPE,MPI_Comm *pcomm);


/**
 * @brief Async the halo message
 * @param cur_imgf       the current image pointer
 * @param nbr_rank       the neighbour's ranks 
 * @param range          the actual size of local image
 * @param rhalo chalo    the customized datatype for swaping halo
 * @param send_req recv_req     send and recv request
 * @param DATATYPE      the customized datatype  * @return double **    new received new edge data vector
**/
void Iswaphalos(double **cur_image, int*nbr_rank, int *range, MPI_Datatype rhalo,  \
               MPI_Datatype chalo, MPI_Comm *pcomm, MPI_Request *send_req, MPI_Request *recv_req);
/**
 * @brief Wait the async tasks 
 * @param nbr_rank       the neighbour's ranks 
 * @param send_req recv_req     send and recv request
 * @param DATATYPE      the customized datatype  * 
 * @return void
**/
void Iwaithalos(int*nbr_rank, MPI_Request *send_req, MPI_Request *recv_req);
/**
 * @brief Calculate the image pixel.
 * @param new_img        the current image pointer
 * @param old_img        the imge updated in last loop. 
 * @param edge           the edge vector
 * @param starti endi start endj    the coordinate of the start and end point.
 * @preturn the max differece
**/
double calculateimg(double **new_img, double **old_img, double **edge, int starti, int startj, int endi, int endj, double *sumnew);