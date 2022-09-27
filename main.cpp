#include <iostream>
#include "mpi.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

using std::cout, std::endl;
void parallel_hello(void){
  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &size);

  cout<<"hello from rank: "<<rank<<"/"<<size<<endl;

  MPI_Finalize();
}

PYBIND11_MODULE(aygalic, m) {
    m.def("parallel_hello", &parallel_hello, "does the thing");
}
