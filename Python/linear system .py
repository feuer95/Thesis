X = np.diag(x)
S = np.diag(s)

Q1 = np.concatenate((np.zeros((c_A,c_A)),A.T,np.eye(c_A)),axis=1)
Q2 = np.concatenate((A,np.zeros((r_A,r_A+c_A))),axis=1)
Q3 = np.concatenate((S,np.zeros((c_A,r_A)),X),axis=1)
Q = np.concatenate((Q1,Q2,Q3),axis=0)

R1 = -(np.dot(A.T,y)) + c
R2 = b - (np.dot(A,x))
R3 = -x*s
R = np.concatenate((R1,R2,R3),axis=0) 

# We compute affine direction

sol = np.dot(np.linalg.inv(Q),R)