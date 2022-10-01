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
  parallel_hello();
  //testing_timer();
  auto t2 = std::chrono::high_resolution_clock::now();
  //end time

  auto diff = std::chrono::duration<double>(t2 - t1);
  cout<<"COMPUTATION TIME: "<<1000*diff.count()<<endl;

  MPI_Finalize();
}
