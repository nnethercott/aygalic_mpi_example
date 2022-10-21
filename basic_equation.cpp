//core
#include <iostream>
#include "mpi.h"
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <type_traits>
#include <tuple>

//misc solvers & linalg stuff
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/index_set.h>

//these are the only ones i really need so far
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>


/*
typedef dealii::TrilinosWrappers::MPI::Vector Vector;
typedef dealii::TrilinosWrappers::SparseMatrix Matrix;
using dealii::IndexSet;
*/

//sparse matrix
void solver_example(){
  dealii::Vector<double> rhs_and_solution(2);

  dealii::SparsityPattern sparsity(2,2,2); //sparsity pattern won't actually give index ownership for (nxm) matrix???? only can use diagonals
  //sparsity.reinit(2,2,2);

  std::cout<<sparsity.max_entries_per_row()<<std::endl;
  std::cout<<sparsity.n_nonzero_elements()<<std::endl;
  std::cout<<sparsity.row_length(0)<<std::endl;
  sparsity.print(std::cout);

  sparsity.compress();
  dealii::SparseMatrix<double> A(sparsity); //create matrix w structure above - REQUIRED
  dealii::SparseDirectUMFPACK solver;

  //create this nice permutation matrix
  A.set(0,0,-1);
  A.set(1,1,-1);

  //arbitrary rhs
  rhs_and_solution[0] = 1;
  rhs_and_solution[1] = -1;

  solver.initialize(A);
  solver.solve(A, rhs_and_solution);

  rhs_and_solution.print(std::cout); //should be [-1 1] since A*[-1 1] = [1 -1]
}

//full matrix - works a lot better since we can actually access all elements in
//the dealii:Matrix (Full instead of Sparse). plus we use the GMRES solver like the
//CoupledReactionDiffusion module in pyfex so results in theory should be comperable
void solver_example2(){
  //problem setup
  dealii::Vector<double> rhs(2);
  dealii::Vector<double> sol(2);
  dealii::FullMatrix<double> A(2,2);

  //populate matrix A
  A.set(0,0,1);
  A.set(0,1,-1);
  A.set(1,0,2);
  A.set(1,1,3);

  //arbitrary rhs
  rhs[0] = 1;
  rhs[1] = 12;

  //solver
  dealii::SolverControl solver_control(1000, 1e-5);
  dealii::PreconditionIdentity preconditioner; //identity preconditioner since problem is mad simple

  dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);

  solver.solve(A, sol, rhs, preconditioner);

  sol.print(std::cout);
}

//setup framework for reading in a matrix(right now we just get the normal dealii matrix working)
dealii::FullMatrix<double> matrix_read(std::string fname){
  std::ifstream ifile(fname);
  if(!ifile){
    std::cerr<<"Matrix file does not exist or failed to open!\n";
    std::exit(1);
  }

  //use second line to get matrix dimensions, we assume we only use real general matrices
  std::string line;
  std::getline(ifile, line);
  std::getline(ifile, line);

  //now we should be at the line with nRows \t nCols \t nnz
  std::istringstream iss(line);
  int nrows, ncols;
  iss>>nrows>>ncols;

  dealii::FullMatrix<double> A(nrows,ncols);

  //need to populate with zeros first (since not sparse)
  for(int i = 0; i<A.m(); i++){
    for(int j = 0; j<A.n(); j++){
      A.set(i, j, 0);
    }
  }


  //now we read in the entries and reinit non zero entries (here we have a very sparse matrix)
  std::size_t i;
  std::size_t j;
  double val;

  while(getline(ifile, line)){
    std::istringstream iss(line);
    iss>>i>>j>>val;
    //we need to revert to 0-indexing
    A.set(i-1, j-1, val);
  }

  return A;
}

//i think this example takes forever to run since we're not taking advantage of matrix sparsity in solver
void solver_example3(){
  std::string path = "/root/lifex_mnt/examples/aygalic_mpi_example/matrices/lnsp_131.mtx";
  dealii::FullMatrix<double> A = matrix_read(path);
  A.print(std::cout);

  //try to use a solver to get the vector we get from passing all 1's through
  dealii::Vector<double> rhs(A.n());
  for(auto it=rhs.begin(); it!=rhs.end(); it++){
    *(it) = 1;
  }

  dealii::Vector<double> b(A.n());
  A.vmult(b, rhs);

  //now lets solve the problem: ans = A*x
  dealii::Vector<double> invrhs(A.n());

  dealii::SolverControl solver_control(1e6, 1e-5);
  dealii::PreconditionIdentity preconditioner; //identity preconditioner

  dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);
  //dealii::SolverCG<dealii::Vector<double>> solver(solver_control); //have a new matrix which is symmetric

  solver.solve(A, invrhs, b, preconditioner);
  invrhs.print(std::cout); //should be all 1's
}

//ohhhhh baby FINALLY WORKS!! WE CAN PROBABLY UP THE SIZE OF THE MATRIX WE'VE IMPORTED
void solver_example4(){
  //--------------------- populate matrix -----------------------
  std::string path = "/root/lifex_mnt/examples/aygalic_mpi_example/matrices/lnsp_131.mtx";
  std::ifstream ifile(path);
  if(!ifile){
    std::cerr<<"Matrix file does not exist or failed to open!\n";
    std::exit(1);
  }

  //use second line to get matrix dimensions, we assume we only use real general matrices
  std::string line;
  std::getline(ifile, line);
  std::getline(ifile, line);

  //now we should be at the line with nRows \t nCols \t nnz
  std::istringstream iss(line);
  int nrows, ncols, nonzero;
  iss>>nrows>>ncols>>nonzero;

  dealii::SparsityPattern sparsity(nrows,ncols,11);

  //we should also store some stuff to not have to go through later
  std::vector<std::tuple<int, int, double>> entries;

  //now we read in the entries and reinit non zero entries (here we have a very sparse matrix)
  int i,j;
  double val;
  while(getline(ifile, line)){
    std::istringstream iss(line);
    iss>>i>>j>>val;
    entries.push_back(std::tuple<int, int, double>(i-1, j-1, val));

    //we need to revert to 0-indexing
    decltype(sparsity.n_rows()) i_ = i-1;
    decltype(sparsity.n_rows()) j_ = j-1;

    sparsity.add(i_,j_);
  }
  //sparsity.print(std::cout);

  sparsity.compress();
  dealii::SparseMatrix<double> A(sparsity);

  //populate A
  for(auto e: entries){
    A.set(std::get<0>(e), std::get<1>(e), std::get<2>(e));
  }

  std::cout<<"("<<A.m()<<" "<<"," << A.n()<<")"<<std::endl;
  std::cout<<"nnz elements: "<<A.n_nonzero_elements()<<std::endl;
  std::cout<<"actually nnz elements: "<<A.n_actually_nonzero_elements()<<std::endl;
  std::cout<<"frobenius norm: "<<A.frobenius_norm()<<std::endl;


  //--------------------- solve linear system -----------------------
  dealii::SparseDirectUMFPACK solver;

  //vector of 1's
  dealii::Vector<double> rhs(A.n());
  for(auto it=rhs.begin(); it!=rhs.end(); it++){
    *(it) = 1;
  }

  dealii::Vector<double> b(A.n());

  A.vmult(b,rhs);
  b.print(std::cout);

  solver.initialize(A);
  solver.solve(A, b);
  b.print(std::cout);
}

int main(){
  solver_example4();
}
