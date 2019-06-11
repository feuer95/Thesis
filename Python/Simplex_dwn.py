"""
~Mathematical Programming~
Simplex implementation.
"""

import numpy as np
from stdForm import stdForm
from input_data import input_data
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type

np.set_printoptions(precision=3, threshold=10, edgeitems=4, linewidth=120)  # Prettier array printing

epsilon = 10**(-10)  # Global truncation threshold



def simplex(A: matrix, b: np.array, c: np.array, rule: int = 0) -> (int, np.array, float, np.array):
    """
    Outer "wrapper" for executing the simplex method: phase I and phase II.

    :param A: constraint matrix
    :param b: independent terms in constraints
    :param c: costs vector
    :param rule: variable selection rule (e.g. Bland's)

    This function prints the outcome of each step to stdout.
    """

    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    """Error-checking"""
    if n < m:
        raise ValueError("Incompatible dimensions "
                         "(no. of variables : {} > {} : no.of constraints".format(n, m))
    if b.shape != (m,):
        raise ValueError("Incompatible dimensions: c_j has shape {}, expected {}.".format(b.shape, (m,)))
    if c.shape != (n,):
        raise ValueError("Incompatible dimensions: c has shape {}, expected {}.".format(c.shape, (n,)))


    "Check full rank matrix"
    if not np.linalg.matrix_rank(A) == m:
        # Remove ld rows:
        A = A[[i for i in range(m) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(n))], :]
        m = A.shape[0]  # Update no. of rows


    """Phase I setup"""
    A[[i for i in range(m) if b[i] < 0]] *= -1  # Change sign of constraints
    b = np.abs(b)  # Idem

    A_I = matrix(np.concatenate((A, np.identity(m)), axis=1))  # Phase I constraint matrix
    x_I = np.concatenate((np.zeros(n), b))  # Phase I variable vector
    c_I = np.concatenate((np.zeros(n), np.ones(m)))  # Phase I c_j vector
    basic_I = set(range(n, n + m))  # Phase I basic variable set


    """Phase I execution"""
    print("Executing phase I...")
    ext_I, x_init, basic_init, z_I, _, it_I = simplex_core(A_I, c_I, x_I, basic_I, rule)
    # ^ Exit code, initial BFS, basis, z_I, d (not needed) and no. of iterations
    print("Phase I terminated.")

    assert ext_I == 0  # assert that phase I has an optimal solution (and is not unlimited)
    if z_I > 0:
        print("\n")
        print_boxed("Unfeasible problem (z_I = {:.6g} > 0).".format(z_I))
        print("{} iterations in phase I.".format(it_I), end='\n\n')
        return 2, None, None, None
    if any(j not in range(n) for j in basic_init):
        # If some artificial variable is in the basis for the initial BFS, exit:
        raise NotImplementedError("Artificial variables in basis")

    x_init = x_init[:n]  # Get initial BFS for original problem (without artificial vars.)

    print("Found initial BFS at x = \n{}.\n".format(x_init))


    """Phase II execution"""
    print("Executing phase II...")
    ext, x, basic, z, d, it_II = simplex_core(A, c, x_init, basic_init, rule)
    print("Phase II terminated.\n")

    if ext == 0:
        print_boxed("Found optimal solution at x =\n{}.\n\n".format(x) +
                    "Basic indexes: {}\n".format(basic) +
                    "Nonbasic indexes: {}\n\n".format(set(range(n)) - basic) +
                    "Optimal cost: {}.".format(z))
    elif ext == 1:
        print_boxed("Unlimited problem. Found feasible ray d =\n{}\nfrom x =\n{}.".format(d, x))

    print("{} iterations in phase I, {} iterations in phase II ({} total).".format(it_I, it_II, it_I + it_II),
          end='\n\n')

    return ext, x, z, d


def simplex_core(A: matrix, c: np.array, x: np.array, basic: set, rule: int = 0) \
        -> (int, np.array, set, float, np.array):
    """
    This function executes the simplex algorithm iteratively until it
    terminates. It is the core function of this project.

    :param A: constraint matrix
    :param c: costs vector
    :param x: initial BFS
    :param basic: initial basic index set
    :param rule: variable selection rule (e.g. Bland's)
    :return: a tuple consisting of the exit code, the value of x, basic index set,
    optimal cost (if optimum has been found), and BFD corresponding to
    feasible ray (if unlimited problem)
    """

    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    assert c.shape == (n,) and x.shape == (n,)  # Make sure dimensions match
    assert isinstance(basic, set) and len(basic) == m and \
           all(i in range(n) for i in basic)  # Make sure that basic is a valid base

    B, N = list(basic), set(range(n)) - basic  # Basic /nonbasic index lists
    del basic  # Let's work in hygienic conditions
    B_inv = inv(A[:, B])  # Calculate inverse of basic matrix (`A[:, B]`)

    z = np.dot(c, x)  # Value of obj. function


    it = 1  # Iteration number
    while it <= 500:  # Ensure procedure terminates (for the min reduced cost rule)
        r_q, q, p, theta, d = None, None, None, None, None  # Some cleanup
        print("\tIteration no. {}:".format(it), end='')


        """Optimality test"""
        prices = c[B] * B_inv  # Store product for efficiency

        if rule == 0:  # Bland rule
            optimum = True
            for q in N:  # Read in lexicographical index order
                r_q = np.asscalar(c[q] - prices * A[:, q])
                if r_q < 0:
                    optimum = False
                    break  # The loop is exited with the first negative r.c.
        elif rule == 1:  # Minimal reduced cost rule
            r_q, q = min([(np.asscalar(c[q] - prices * A[:, q]), q) for q in N],
                         key=(lambda tup: tup[0]))
            optimum = (r_q >= 0)
        else:
            raise ValueError("Invalid pivoting rule")

        if optimum:
            print("\tfound optimum")
            return 0, x, set(B), z, None, it  # Found optimal solution


        """Feasible basic direction"""
        d = np.zeros(n)
        for i in range(m):
            d[B[i]] = trunc(np.asscalar(-B_inv[i, :] * A[:, q]))
        d[q] = 1


        """Maximum step length"""
        # List of tuples of "candidate" theta an corresponding index in basic list:
        neg = [(-x[B[i]] / d[B[i]], i) for i in range(m) if d[B[i]] < 0]

        if len(neg) == 0:
            print("\tidentified unlimited problem")
            return 1, x, set(B),  None, d, it  # Flag problem as unlimited and return ray

        # Get theta and index (in basis) of exiting basic variable:
        theta, p = min(neg, key=(lambda tup: tup[0]))


        """Variable updates"""
        x = np.array([trunc(var) for var in (x + theta * d)])  # Update all variables
        assert x[B[p]] == 0

        z = trunc(z + theta * r_q)  # Update obj. function value

        # Update inverse:
        for i in set(range(m)) - {p}:
            B_inv[i, :] -= d[B[i]]/d[B[p]] * B_inv[p, :]
        B_inv[p, :] /= -d[B[p]]

        N = N - {q} | {B[p]}  # Update nonbasic index set
        B[p] = q  # Update basic index list

        """Print status update"""
        print(
            "\tq = {:>2} \trq = {:>9.2f} \tB[p] = {:>2d} \ttheta* = {:>5.4f} \tz = {:<9.2f}"
                .format(q + 1, r_q, B[p] + 1, theta, z)
        )

        it += 1


    # If loop goes over max iterations (500):
    raise TimeoutError("Iterations maxed out (probably due to an endless loop)")


def print_boxed(msg: str) -> None:
    """
    Utility for printing pretty boxes.
    :param msg: message to be printed
    """

    lines = msg.splitlines()
    max_len = max(len(line) for line in lines)

    if max_len > 100:
        raise ValueError("Overfull box")

    print('-' * (max_len + 4))
    for line in lines:
        print('| ' + line + ' ' * (max_len - len(line)) + ' |')
    print('-' * (max_len + 4))



def trunc(x: float) -> float:
    """
    Returns 0 if x is smaller (in absolute value) than a certain global constant.
    """
    return x if abs(x) >= epsilon else 0

#(A, b, c) = input_data(3)
#(A, c) = stdForm(A, c)
#A = np.asmatrix(A)
#x = simplex(A, b, c)
