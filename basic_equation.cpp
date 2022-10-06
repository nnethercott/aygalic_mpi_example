//core
#include <iostream>
#include "mpi.h"
#include <vector>

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

  //create this nice permutation matrix
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

  sol.print(std::cout); //should be [-1 1] since A*[-1 1] = [1 -1]
}


int main(){
  solver_example2();
}
