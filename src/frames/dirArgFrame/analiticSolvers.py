import sympy as sp
alpha1, alpha2 = sp.symbols('alpha_1 alpha_2')
n = 2

psi1, psi2 = sp.symbols('psi_1 psi_2')
b1, b2 = sp.symbols('b_1 b_2')
x1, x2 = sp.symbols('x_1 x_2')
v1, v2,v3,v4 =sp.symbols('v_1:5')
u = psi1*b1+psi2*b1/(2*alpha2)
a11, a22, a21, a12 = sp.symbols('a_{11}, a_{22} a_{21} a_{12}')
M = sp.Matrix([[a11, a12, b1**2/(2*alpha2), b2*b1/(2*alpha2)],
               [a21, a22, b1*b2/(2*alpha2), b2**2/(2*alpha2)],
               [0,0,a11,0],
               [0,0,0,a22]])
for k, v in zip(M.eigenvals(), M.eigenvects()):
    print(k, v)
    pass
