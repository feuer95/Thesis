"""
Created on Thu Apr 18 16:52:35 2019

@author: Elena
"""
from starting_point import sp # Function to find the initial infeasible point
from print_boxed import print_boxed # Print pretty info boxes
from stdForm import stdForm # Function to extend LP in a standard form
import numpy as np # To create vectors
from input_data import input_data
from cent_meas import cent_meas
from term import term 

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''                                 ====
                      PREDICTOR-CORRECTOR MEHROTRA ALGORITHM 
                                    ====

Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            c_form: canonical form -> 0 by default
            
            
'''

def mehrotra(A, b, c, c_form = 0, w = 10**(-8), max_iter = 500):
    
    print('\n\tCOMPUTATION OF MEHROTRA ALGORITHM\n')
    
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
        raise Exception('Inputs must be a numpy arrays')
    
    # Input data in a standard form [A | I]:
    if c_form == 0:
        A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    
    """ Check full rank matrix """
    
    if not np.linalg.matrix_rank(A) == r_A: # Remove ld rows:
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows

    """ Initial points """
    
    # Initial infeasible positive (x,y,s) and Initial gap g
    (x,y,s) = sp(A, c, b)    
    g = np.dot(c,x) - np.dot(y,b)
    
    print('\nInitial primal-dual point:\nx = {} \nlambda = {} \ns = {}'.format(x, y, s))    
    print('Dual initial g: {}.\n'.format("%10.3f"%g))      
  
    #%%
    
    """ Predictor step: compute affine direction """
    
    # Compute affine scaling direction solving Qs = R with an augmented system D^2 = S^{-1}*X 
    # with Q a large sparse matrix 
    # and R = [rb, rc, - x_0*s_0]
    
    it = 0
    tm = term(it)
    u = []
    u.append([it, g, x, s, b - np.dot(A,x), c - np.dot(A.T, y) - s])
    while tm > w:
        print("\n\tIteration: {}\n".format(it), end='')   
        # CHOLESKY for normal equation with matrix A* D^2 *A^{T}
        S_inv = np.linalg.inv(np.diag(s))  # S^{-1}
        W1 = np.dot(S_inv, np.diag(x))     # W1 = D = S^(-1)*X    
        W2 = np.dot(A, W1)                 # W2      A*S^(-1)*X
        W  = np.dot(W2, A.T)
        L = np.linalg.cholesky(W) 
        L_inv = np.linalg.inv(L)
        
        # RHS of the system, including the minus
        
        rb = b - np.dot(A, x) 
        rc = c - np.dot(A.T, y) - s
        rxs = - x*s
        
        B = rb + np.dot(W2, rc) - np.dot(np.dot(A, S_inv), rxs) #RHS of normal equation form
        z = np.dot(L_inv, B)
        
        # SEARCH DIRECTION:
        y1 = np.dot(L_inv.T, z)
        s1 = rc - np.dot(A.T, y1)
        x1 = np.dot(S_inv, rxs) - np.dot(W1, s1)
        print("\nPREDICTOR STEP:\nAffine direction:\n({}, {}, {})\n".format(x1, y1, s1))
        
        #%%
        
        """ Centering step: compute search direction """
        
        # Find alfa1_affine and alfa2_affine : maximum steplength along affine-scaling direction
        # Find the minimum( x_{i}/x1_{i} , 1) and the minimum( s_{i}/s1_{i} , 1)
        
        h = min([(-x[i]/ x1[i], i) for i in range(c_A) if x1[i] < 0], default = [0])[0]
        k = min([(-s[i]/ s1[i], i) for i in range(c_A) if s1[i] < 0], default = [0])[0]
        
        alfa1 = min(h,1)
        alfa2 = min(k,1)
        
        # Set the centering parameter to Sigma = (mi_aff/mi)^{3}   
        mi = np.dot(x,s)/c_A # Duality measure
        mi_af = np.dot(x + alfa1*x1,s + alfa2*s1)/c_A # Average value of the incremented vectors
        Sigma = (mi_af/mi)**3
        
        # SECOND SEARCH DIRECTION
        Rxs = - x1*s1 + Sigma*mi*np.ones((c_A)) # RHS of the New system, including minus
        
        Rb = - np.dot(np.dot(A,S_inv), Rxs) # RHS of New normal equation form
        z = np.dot(L_inv, Rb)
        
        # SEARCH DIRECTION:
        y2 = np.dot(L_inv.T, z)
        s2 = - np.dot(A.T, y2)
        x2 = np.dot(S_inv, Rxs) - np.dot(W1, s2)
        print("\nPREDICTOR STEP:\nAffine direction:\n({}, {}, {})\n".format(x2, y2, s2))
 
        #%%
        
        """ Corrector step: compute (x_k+1, lambda_k+1, s_k+1) """
        
        # The steplength without eta_k
        x2 += x1
        s2 += s1
        y2 += y1
        H = min([(-x[i]/ x2[i], i) for i in range(c_A) if x2[i] < 0], default = [0])[0]
        K = min([(-s[i]/ s2[i], i) for i in range(c_A) if s2[i] < 0], default = [0])[0]
        Alfa1 = min(0.99*H, 1)
        Alfa2 = min(0.99*K, 1)
        
        # Compute the final update , using sol2 and Alfa_i 
        
        x += Alfa1*x2
        y += Alfa2*y2
        s += Alfa2*s2
        it += 1
        if it == max_iter:
           return x, s, u 
           raise TimeoutError("Iterations maxed out")

        # Dual gap c^{T}*x - b^{T}*y = x*s
        z = np.dot(c,x)
        g = z - np.dot(y,b)
                
        # Termination elements
        tm = term(it, b, c, rb, rc, z, g)
        print('Dual next gap: {}.\n'.format("%10.3f"%g))

        u.append([it, g, x.copy(), s.copy(), rb.copy(), rc.copy()])
        print('CORRECTOR STEP:\nCurrent primal-dual point: \n x_k = ',x,'\n s_k = ',s,'\nl_k = ',y)
        print('Current g: {}\n'.format("%.3f"%g))        
        
    print_boxed("Found optimal solution of the standard problem at\n x* = {}.\n\n".format(x) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    return x, s, u 
     

#%%ù
    
"""Input data"""

# Input data of canonical LP:
if __name__ == "__main__":
    
    # Input data of canonical LP:
    (A, b, c) = input_data(2)
    
    xp, s, u = mehrotra(A, b, c, max_iter = 7)
    
    cent_meas(xp, u, ' Mehrotra')

