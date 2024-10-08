#!/usr/bin/env python
"""
A simple benchmark script to compare the performance of the original Corrfunc and pycorrfunc DD functions.

Run "python bench.py --help" for more information.
"""

import os

import click
import timeit
import numpy as np
from Corrfunc.theory.DD import DD as corrfunc_DD
from pycorrfunc.theory import DD as pycorrfunc_DD

NTHREAD = len(os.sched_getaffinity(0))

BOXSIZE = 1.0


@click.command()
@click.option(
    '-N',
    '--Npts',
    'N',
    default=10**5,
    type=int,
    help='Number of points to generate for the benchmark.',
)
@click.option('-b', '--bins', default=10, help='Number of bins for the DD function.')
@click.option('-d', '--dtype', default='f8', help='Data type to use for the benchmark.')
@click.option(
    '-t',
    '--nthread',
    default=NTHREAD,
    help='Number of threads to use for the benchmark.',
)
@click.option(
    '-i', '--isa', default='avx', help='Instruction set to use for the benchmark.'
)
@click.option('-v', '--verbose', is_flag=True, help='Print verbose output.')
@click.option(
    '-w', '--weighted', is_flag=True, help='Use weighted points for the benchmark.'
)
def benchmark(N, bins, dtype, nthread, isa, verbose, weighted):
    """
    Benchmark original Corrfunc and pycorrfunc DD.
    """
    # Generate random points
    rng = np.random.default_rng(1237)
    pos = rng.random(size=(3, N), dtype=dtype) * BOXSIZE

    w = rng.random(size=N, dtype=dtype) if weighted else None

    # Define bin edges
    bin_edges = np.linspace(0, BOXSIZE / 5, bins + 1)

    # Benchmark Corrfunc
    corrfunc_isa = 'avx512f' if isa == 'avx512' else isa

    def corrfunc_benchmark(**kwargs):
        return corrfunc_DD(
            1,
            nthread,
            bin_edges,
            *pos,
            weights1=w,
            boxsize=BOXSIZE,
            periodic=True,
            isa=corrfunc_isa,
            output_ravg=weighted,
            **kwargs,
        )

    corrfunc_benchmark(verbose=True)

    nrep, corrfunc_time = autorange(timeit.Timer(corrfunc_benchmark))
    corrfunc_time /= nrep

    # Benchmark pycorrfunc
    def pycorrfunc_benchmark(**kwargs):
        return pycorrfunc_DD(
            *pos,
            bin_edges,
            W1=w,
            boxsize=BOXSIZE,
            num_threads=nthread,
            isa=isa,
            dtype=dtype,
            do_ravg=weighted,
            **kwargs,
        )

    pycorrfunc_benchmark(verbose=True)

    nrep, pycorrfunc_time = autorange(timeit.Timer(pycorrfunc_benchmark))
    pycorrfunc_time /= nrep

    # Print result
    print(f'Corrfunc DD:   {corrfunc_time:.3f} s')
    print(f'pycorrfunc DD: {pycorrfunc_time:.3f} s')

    # Print config
    print(f'{N=}, {bins=}, {dtype=}, {nthread=}')


def autorange(timer: timeit.Timer, min_time=2.0):
    """Adapted from timeit.Timer.autorange"""
    i = 1
    while True:
        for j in 1, 2, 5:
            number = i * j
            time_taken = timer.timeit(number)
            if time_taken >= min_time:
                return (number, time_taken)
        i *= 10


if __name__ == '__main__':
    benchmark()
