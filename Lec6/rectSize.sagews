from scipy import optimize
import numpy as np

from itertools import chain, combinations

A = Matrix([[0,0,1,1,1,1],
            [1,0,0,0,1,1],
            [0,1,1,1,0,0],
            [1,1,0,1,1,0],
            [1,1,1,0,1,0]])

# Grab all coordinates with required value
def index_map(Mf, value):
    return dict((v,k) for (k,v) in enumerate([(row,col) for row in range(Mf.nrows()) for col in range(Mf.ncols()) if Mf[row][col] == value]))

# Generate all possible rectangles
def all_rects(rowcount, colcount):
    for rows in chain.from_iterable(combinations(range(rowcount), r) for r in range(1, rowcount+1)):
        for cols in chain.from_iterable(combinations(range(colcount), c) for c in range(1, colcount+1)):
            yield [(r,c) for r in rows for c in cols]

# Generate all monochromatic rectangles
def all_mono_rects(Mf, value):
    for R in all_rects(Mf.nrows(), Mf.ncols()):
        if all((Mf[x][y] == value for (x,y) in R)):
            yield R

# Compute the max(mu(R)) over all monochromatic rectangles
def maxMuR(mu, Mf, id_map, value):
    return max(sum(mu[id_map[coord]] for coord in R) for R in all_mono_rects(Mf, value))

# Return the max weight rectangle with the distribution
def getMaxWeightR(mu, Mf, id_map, value):
    maxmuR = maxMuR(mu, Mf, id_map, value)
    for R in all_mono_rects(Mf, value):
        if sum(mu[id_map[coord]] for coord in R) == maxmuR:
            return [(coord, mu[id_map[coord]]) for coord in R if mu[id_map[coord]] != 0]

# Compute the RS(f) for value, given Mf
def rectSize(Mf, value):
    mu_id_map = index_map(Mf, value)
    mu = optimize.minimize(maxMuR,
                           np.random.rand(len(mu_id_map)),
                           args=(Mf, mu_id_map, value),
                           bounds=[(0,1)]*(len(mu_id_map)),
                           constraints=({'type':'eq', 'fun': lambda mu: sum(mu)-1}),
                           options={'ftol': 1e-12, 'maxiter': 4096})
    # Return a matrix of mu distribution
    return (Matrix(np.around([[0 if cell != value else mu.x[mu_id_map[(r,c)]] for (c,cell) in enumerate(row)] for (r,row) in enumerate(Mf.rows())], 3)),
            getMaxWeightR(np.around(mu.x,3), A, mu_id_map, 1),
            mu.fun)

rectSize(A,1)
