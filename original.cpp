#include <iostream>
#include "mpi.h"
#include <chrono>
#include <thread>

using std::cout, std::endl;
void parallel_hello(void){
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &size);

  cout<<"hello from rank: "<<rank<<"/"<<size<<endl;
}

void testing_timer(){ //output from my main should be around 1000
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

int main(){
  MPI_Init(NULL, NULL);

  //timing the function
  auto t1 = std::chrono::high_resolution_clock::now();
  for(auto i = 0; i<100; i++){
    parallel_hello();
  }
  //testing_timer();
  auto t2 = std::chrono::high_resolution_clock::now();
  //end time

  auto diff = std::chrono::duration<double>(t2 - t1);
  double elapsed = 1000*diff.count();

  //what if we reduced and took the maximal runtime
  double max_elapsed;
  MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  //seems inneficient to call this again
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0) cout<<"MAX COMPUTATION TIME: "<<max_elapsed<<endl;

  MPI_Finalize();
}
