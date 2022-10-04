"""
we copy the structure of the dealii step 40 tutorial of
1. setup_system
2. assemble_system
3. solve

and time for x number of iterations

CONCERNS: maybe our last solution is close to convergence and our sim times decay super quick 
"""


# Imports
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#do we need this?
import sys,os
sys.path.insert(1, '/root/lifex_mnt/pyfex/pyfex/utils/numerics')
from RB_library import *

from arrayBind import *
from CoupledReactionDiffusion import *


d = CoupledReactionDiffusion()
d.model = "reaction_diffusion" # reaction_diffusion | diffusion
d.refinement_reaction_diffusion = 4

d.make_grid()

def run():
    d.setup_system()
    d.assemble_system()
    d.solve_system()

    #try to extract the solution, currently using the original arrayBind lib
    #d_solution = Vector(d.solution) #Vector(const Vector &) copy constr.
    #print(f'size of solution is: {d_solution.size()}')

n_iters = 100

t1 = MPI.Wtime()
for i in range(n_iters):
    run()
    #time.sleep(1)
t2 = MPI.Wtime()

diff = np.array(1000*(t2-t1))
max_elapsed = np.zeros(1)

comm.Barrier()
comm.Reduce([diff, MPI.DOUBLE], [max_elapsed, MPI.DOUBLE], op=MPI.MAX, root=0)

if rank == 0:
    print(f'COMPUTATION TIME: {max_elapsed}')
