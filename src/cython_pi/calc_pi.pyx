# cython: language_level=3
import cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time as c_time

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_pi_cython(int n):
    cdef:
        int i, h = 0
        double x, y

    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Seed C RNG once per call (simple approach)
    srand(<unsigned int> c_time(NULL))

    for i in range(n):
        x = rand() / <double>RAND_MAX
        y = rand() / <double>RAND_MAX

        if x*x + y*y <= 1.0:
            h += 1

    return 4.0 * h / n
