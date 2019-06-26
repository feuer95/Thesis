"""
Created on Thu Apr 18 16:52:35 2019

@author: Elena
"""
from starting_point import sp       # Function to find the initial infeasible point
from print_boxed import print_boxed # Print pretty info boxes
from stdForm import stdForm         # Function to extend LP in a standard form
import numpy as np                  # To create vectors
from input_data import input_data
from cent_meas import cent_meas
from term import term 

# Clean form of printed vectors
np.set_printoptions(precision = 4, threshold = 10, edgeitems = 4, linewidth = 120, suppress = True)


'''                                 ====
                      PREDICTOR-CORRECTOR MEHROTRA ALGORITHM 
                                    ====

Input data: np.arrays of matrix A, cost vector c, vector b of the LP
            c_form: canonical form -> default 0
            w = tollerance -> default 10^(-8)
            
Output data: x: primal solution
             s: dual solution
             u: list: iteration, dual gas, Current x, Current s, Feasibility x, Feasibility s

Matricial system computed with normal equations
'''

def mehrotra(A, b, c, c_form = 0, w = 10**(-8), max_it = 500):
    
    print('\n\tCOMPUTATION OF MEHROTRA ALGORITHM\n')
        
    # Algorithm in 4 steps:  
    # 0..Input error checking
    # 1..Find the initial point with Mehrotra's method 
    # 2..Predictor step
    # 3..Centering step
    # 4..Corrector step
    
    """ Input error checking & construction in a standard form """
        
    if not (isinstance(A, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray)):
       raise Exception('Inputs must be a numpy arrays')
        
    if c_form == 0:
        A, c = stdForm(A, c)    
    r_A, c_A = A.shape
    if not np.linalg.matrix_rank(A) == r_A: # Check full rank matrix:Remove ld rows:
        A = A[[i for i in range(r_A) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(c_A))], :]
        r_A = A.shape[0]  # Update no. of rows
    
    """ Initial points: Initial infeasible positive (x,y,s) and initial gap g """
    
    (x, y, s) = sp(A, c, b)    
    g = np.dot(c, x) - np.dot(y, b)
    
    print('\nInitial primal-dual point:\nx = {} \nlambda = {} \ns = {}'.format(x, y, s))    
    print('Dual initial g: {}.\n'.format("%10.3f"%g))      
  
    #%%
    
    """ Predictor step: compute affine direction """
    '''
    Compute affine scaling direction solving Qs = R with an augmented system D^2 = S^{-1}*X 
    with Q a large sparse matrix 
    and R = [rb, rc, - x_0*s_0]
    '''
    
    it = 0        # Num of iterations
    tm = term(it) # Tollerance in the cycle
    
    u = []        # Construct list of info elements 
    sig = []
    u.append([it, g, x, s, b - np.dot(A,x), c - np.dot(A.T, y) - s])
    
    while tm > w:
        
        print("\n\tIteration: {}\n".format(it), end='')   
        
        """ Pure Newton's method with with normal equations: find the direction vector (y1, s1, x1)"""
        
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
        
        # SEARCH DIRECTION affine:
        y1 = np.dot(L_inv.T, z)
        s1 = rc - np.dot(A.T, y1)
        x1 = np.dot(S_inv, rxs) - np.dot(W1, s1)
        print("\nPREDICTOR STEP:\nAffine direction:\n({}, {}, {})\n".format(x1, y1, s1))
        
        #%%
        
        """ Centering step: compute search direction """
        '''
        Find alfa1_affine and alfa2_affine : maximum steplength along affine-scaling direction
        Find the minimum( x_{i}/x1_{i} , 1) and the minimum( s_{i}/s1_{i} , 1)
        '''
        
        h = min([(-x[i]/ x1[i], i) for i in range(c_A) if x1[i] < 0], default = [0])[0]
        k = min([(-s[i]/ s1[i], i) for i in range(c_A) if s1[i] < 0], default = [0])[0]
        
        alfa1 = min(h,1)
        alfa2 = min(k,1)
        
        # Set the centering parameter to Sigma = (mi_aff/mi)^{3}   
        mi = np.dot(x,s)/c_A # Duality measure
        mi_af = np.dot(x + alfa1*x1,s + alfa2*s1)/c_A # Average value of the incremented vectors
        Sigma = (mi_af/mi)**(3)
        
        # SECOND SEARCH DIRECTION
        Rxs = - x1*s1 + Sigma*mi*np.ones((c_A)) # RHS of the New system, including minus
        
        Rb = - np.dot(np.dot(A,S_inv), Rxs) # RHS of New normal equation form
        z = np.dot(L_inv, Rb)
        
        # SEARCH DIRECTION centering-orrector:
        y2 = np.dot(L_inv.T, z)
        s2 = - np.dot(A.T, y2)
        x2 = np.dot(S_inv, Rxs) - np.dot(W1, s2)
        print("Cc direction:\n({}, {}, {})\n".format(x2, y2, s2))
 
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
        
        # Update        
        x += Alfa1*x2            # Current x
        y += Alfa2*y2            # Current y
        s += Alfa2*s2            # Current s
        z = np.dot(c, x)         # Current optimal solution
        g = z - np.dot(y, b)     # Current gap 
        it += 1
        u.append([it, g, x.copy(), s.copy(), rb.copy(), rc.copy()])
        sig.append([Sigma, mi_af, mi]) 

        # Termination elements
        tm = term(it, b, c, rb, rc, z, g)

        if it == max_it:
           raise TimeoutError("Iterations maxed out")


        print('CORRECTOR STEP:\nCurrent primal-dual point: \n x = ',x,'\nlambda = ',y,'\n s = ',s)
        print('Current g: {}\n'.format("%.3f"%g))        

#%%
        
    print_boxed("Found optimal solution of the standard problem at\n x* = {}.\n\n".format(x) +
                "Dual gap: {}\n".format("%10.6f"%g) +
                "Optimal cost: {}\n".format("%10.3f"%z) +
                "Number of iteration: {}".format(it))
    return x, s, u
     

#%%
'''                           ===
                   MEHROTRA METHOD test
                              ===
'''
if __name__ == "__main__":
    
#    (A, b, c) = input_data(1)
    
    xm, sm, um = mehrotra(A, b, c)
    
    dm = cent_meas(xm, um, 'Mehrotra', plot = 0)

