```python
import cvxpy as cp
import numpy as np
```


```python
m = 15
n = 10
```


```python
np.random.seed(1)
s0 = np.random.random(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
```

    
    The optimal value is 0.0
    A solution x is
    [-0.73494648  1.65760314 -1.36536879  0.10218965  0.79236377 -0.36227787
      0.82022999  0.48476277  0.28663826 -0.46552217]
    


```python
A1 = np.array([[1, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 1]])
b1 = np.array([400, 600, 500])
```


```python
A2 = np.array([[0.4, 1.1, 1, 0, 0, 0],
              [0, 0, 0, 0.5, 1.2, 1.3]])
b2 = np.array([800, 900])
```


```python
A3 = np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1]])
b3 = np.array([0, 0, 0, 0, 0, 0])
```


```python
c = np.array([13, 9, 10, 11, 12, 8])
```


```python
n = 6
x = cp.Variable(n)
```


```python
prob = cp.Problem(cp.Minimize(c.T @ x),
                  [A1 @ x == b1, A2 @ x <= b2, A3 @ x >= b3])
prob.solve()
```




    13800.00000202733




```python
print("\n目标函数的最小值", prob.value)
print("所求的x矩阵")
print(x.value)
```

    
    目标函数的最小值 13800.00000202733
    所求的x矩阵
    [1.85735461e-07 6.00000000e+02 1.50597739e-07 4.00000000e+02
     4.51551936e-07 5.00000000e+02]
    


```python
A1 = np.array([[2, 1],
              [1, 2]])
b1 = np.array([8, 10])
```


```python
A2 = np.array([[1, 0],
               [0, 1]])
b2 = np.array([0, 0])
```


```python
c = np.array([1, 2])
n = 2
x = cp.Variable(n)
```


```python
prob = cp.Problem(cp.Maximize(c.T @ x),
                  [A1 @ x <= b1, A2 @ x >= b2])
prob.solve()
```




    9.99999998623802




```python
print(prob.value)
print(x.value)
```

    9.99999998623802
    [1.32678804 4.33660597]
    


```python
"""
Created on Sun Sep 26 10:21:41 2021
@author: songyuhui
function： 利用scipy实现最优化问题
        eg：目标函数：max z = 2x1 + 3x2- 5x3
            约束条件： x1 + x2+ x3 = 7
                     2x1 - 5x2 + x3 >= 10
                     x1 + 3x2 + x3 <= 12
                     x1,x2,x3 >=0
        用optimize库在求解时全部都要转化成标准形式：
        目标函数：min c^T x 
        约束条件： Ax <= B
                  Aeq*x =Beq
                  LB<= x <=UB
        求解代码为：
        求解函数：res=optimize.linprog(C,A,B,Aeq,Beq,LB,UB,Xo,options)
        结果：         res
        目标函数最小值：res.fun 
        最优解         res.x
        
"""
from scipy import optimize
import numpy as np

C=np.asarray([2, 3, -5])
A=np.asarray([[-2, 5, 1], [1, 3, 1]])
B=np.asarray([-10, 12])
Aeq=np.asarray([[1, 1, 1]])
Beq=np.asarray([7])

res=optimize.linprog(-C,A,B,Aeq,Beq)
print(res)
```

         con: array([6.55013821e-09])
         fun: -14.571428552851675
     message: 'Optimization terminated successfully.'
         nit: 5
       slack: array([-1.61912528e-08,  3.85714286e+00])
      status: 0
     success: True
           x: array([6.42857142e+00, 5.71428572e-01, 7.96679179e-10])
    


```python
C=np.asarray([1, 2])
A=np.asarray([[2, 1], [1, 2], [-1, 0], [0, -1]])
B=np.asarray([8, 10, 0, 0])
res=optimize.linprog(-C, A, B)
print(res)
```

         con: array([], dtype=float64)
         fun: -9.999999999940046
     message: 'Optimization terminated successfully.'
         nit: 4
       slack: array([1.17568983e+00, 5.99538197e-11, 1.21620678e+00, 4.39189661e+00])
      status: 0
     success: True
           x: array([1.21620678, 4.39189661])
    


```python
A1 = np.array([[1, 1, 0， 0], [1, 0, 0， 1], [1, 0, 0, 1]])
B1 = np.array([1, 1, 1, 1])
C = np.array([3, 1, 4, 4])
n = 4
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(C.T @ x),
                [A1 @ x >= B1])
prob.solve()
print("cvxpy minimum value:", prob.value)
print("Point:", x.value)
```

    cvxpy minimum value: 2.524073922602572e-10
    Point: [2.52407392e-10 6.71020412e-01 6.71020412e-01]
    


```python
C = np.asarray([1, 0, 0])
A = np.asarray([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
B = np.asarray([0, 0, 0])
res=optimize.linprog(C, A, B)
print("Scipy minimum value:",res.fun)
print("Point:", res.x)
```

    Scipy minimum value: 0.0
    Point: [0. 0. 0.]
    


```python
import cvxpy as cp
import numpy as np
import scipy.io as scio
import time
x = cp.Variable(2)
A = np.array([[2, 1], [1, 2]])
B = np.array([1, 1])
C = np.array([2, 1])
prob = cp.Problem(cp.Minimize(C.T @ x),
                 [A @ x >= B, x>=0])
prob.solve()
start = time.time()
prob.solve()
end = time.time()
#print("cvxpy time:", end - start, "s")
print("cvxpy minimum value: ", prob.value)
print("Point: ", x.value)
```

    cvxpy minimum value:  0.9999999998991901
    Point:  [0.2358356 0.5283288]
    


```python
import cvxpy as cp
import numpy as np 
import scipy.io as scio
import time
data= scio.loadmat('C:\instance_large.mat')
A = np.asarray(data['A'])
B = np.asarray(data['B'])
C = np.asarray(data['C'])
C = np.squeeze(C)
n = A.size
#E = np.eye(n)
#zero = np.zeros(n)
x = cp.Variable(n)
#cp.constraints.nonpos.NonPos(-x)
prob = cp.Problem(cp.Maximize(A @ x),
                 [B @ x <= C, x >= 0])
#E @ x >= zero]
start = time.time()
prob.solve()
end = time.time()
print("cvxpy time:", end - start, "s")
print("cvxpy maximum value:", prob.value)
#print("Point:", x.value)
#python_y=np.array(data['matlab_y'])
```

    cvxpy time: 3344.1226658821106 s
    cvxpy maximum value: 54087865.83065152
    


```python
import scipy.io as scio
from scipy.optimize import linprog
import time
data = scio.loadmat('C:\instance_large.mat', mat_dtype=True)
A = np.array(data['A'])[0]
n = A.size
for i in range(n):
    A[i] = -A[i]
B = np.array(data['B'])
C = np.array(data['C'])
C = np.squeeze(C)

#Z = np.zeros(100)
zero = ((0, None),)*n
#none = np.array(none)
#print(zero)
start = time.time()
res=linprog(A, A_ub=B, b_ub=C, bounds=zero)
end = time.time()
print("Scipy time:", end - start, "s")
#print(-res.fun)
print("Scipy maximum value", -res.fun) 
```

    Scipy time: 208.4388017654419 s
    Scipy maximum value 54087865.834705986
    


```python
from pulp import *
import scipy.io as scio
import time
prob = LpProblem('max_z', sense=LpMaximize)
data = scio.loadmat('C:\instance_medium.mat', mat_dtype=True)
A = np.array(data['A'])[0]
B = np.array(data['B'])
C = np.array(data['C'])
C = np.squeeze(C)
n = A.size
x = [LpVariable(f"x{i}", lowBound=0, upBound=None, cat='LpContinuous') for i in range(0, n)]
num = B.shape[0]
prob += np.sum(np.array(A)*np.array(x))
for i in range(num):
    prob += np.sum(np.array(B[i])*np.array(x)) <= C[i]

start = time.time()
status = prob.solve()
end = time.time()
print("PuLp time:", end-start, "s")
print(f"PuLp maximum value:{value(prob.objective)}")
#print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x1 = LpVariable('x1', 0, 1, cat='LpContinuous')
x2 = LpVariable('x2', 0, 1, cat='LpContinuous')
x3 = LpVariable('x3', 0, 1, cat='LpContinuous')
x4 = LpVariable('x4', 0, 1, cat='LpContinuous')

# 设置目标函数
prob += x1+x2+x3+x4
# 约束条件
prob += x1+x2>=1
prob += x1+x3>=1
prob += x1+x4>=1
prob += x2+x3>=1
prob += x3+x4>=1

#print(prob)
status = prob.solve(use_mps=True)
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:2.0
    Point: {'x1': 0.5, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5}
    


```python
import cvxpy as cp
import numpy as np 
import scipy.io as scio
import time

A1 = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
B1 = np.array([1, 1, 1])
C = np.array([3, 2, 1, 1])
n = 4
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(C.T @ x),
                [A1 @ x >= B1, x>=0, x<=1])
prob.solve()
print("cvxpy minimum value:", prob.value)
print("Point:", x.value)
```

    cvxpy minimum value: 3.0000000001708313
    Point: [1.00000000e+00 2.71568851e-10 4.27090495e-10 4.27090495e-10]
    


```python
import cvxpy as cp
import numpy as np

A1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1], 
               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1], 
               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
B1 = np.array([0, 0, 0])
A2 = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
               [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
B2 = np.array([3, 2, 5, 10, 5])
C = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
x = cp.Variable(16)
prob = cp.Problem(cp.Minimize(C @ x),
                 [A1 @ x <= B1, A2 @ x == B2, x>=0])
prob.solve()
print(prob.value)
print(x.value)
```

    8.333333331498833
    [1.         0.66666667 1.66666667 3.33333333 1.66666667 1.
     0.66666667 1.66666667 3.33333333 1.66666667 1.         0.66666667
     1.66666667 3.33333333 1.66666667 8.33333333]
    


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x11 = LpVariable('x11', 0, None, cat='LpContinuous')
x12 = LpVariable('x12', 0, None, cat='LpContinuous')
x13 = LpVariable('x13', 0, None, cat='LpContinuous')
x21 = LpVariable('x21', 0, None, cat='LpContinuous')
x22 = LpVariable('x22', 0, None, cat='LpContinuous')
x23 = LpVariable('x23', 0, None, cat='LpContinuous')
x31 = LpVariable('x31', 0, None, cat='LpContinuous')
x33 = LpVariable('x33', 0, None, cat='LpContinuous')
x34 = LpVariable('x34', 0, None, cat='LpContinuous')
x35 = LpVariable('x35', 0, None, cat='LpContinuous')
x44 = LpVariable('x44', 0, None, cat='LpContinuous')
x45 = LpVariable('x45', 0, None, cat='LpContinuous')
L = LpVariable('L', 0, None, cat='LpContinuous')
# 设置目标函数
prob += L
# 约束条件
prob += x11+x21+x31==2
prob += x12+x22==2
prob += x13+x23+x33==2
prob += x34+x44==2
prob += x35+x45==2
prob += x11+x12+x13-L<=0
prob += x21+x22+x23-L<=0
prob += x31+x33+x34+x35-L<=0
prob += x44+x45-L<=0
#print(prob)
status = prob.solve()
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:2.5
    Point: {'L': 2.5, 'x11': 2.0, 'x12': 0.0, 'x13': 0.5, 'x21': 0.0, 'x22': 2.0, 'x23': 0.5, 'x31': 0.0, 'x33': 1.0, 'x34': 1.5, 'x35': 0.0, 'x44': 0.5, 'x45': 2.0}
    


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x11 = LpVariable('x11', 0, None, cat='LpContinuous')
x12 = LpVariable('x12', 0, None, cat='LpContinuous')
x13 = LpVariable('x13', 0, None, cat='LpContinuous')
x22 = LpVariable('x22', 0, None, cat='LpContinuous')
x23 = LpVariable('x23', 0, None, cat='LpContinuous')
x24 = LpVariable('x24', 0, None, cat='LpContinuous')
x32 = LpVariable('x32', 0, None, cat='LpContinuous')
x33 = LpVariable('x33', 0, None, cat='LpContinuous')
L = LpVariable('L', 0, None, cat='LpContinuous')
# 设置目标函数
prob += L
# 约束条件
prob += x11==3
prob += x12+x22+x32==6
prob += x13+x23+x33==6
prob += x24==3
prob += x11+x12+x13-L<=0
prob += x22+x23+x24-L<=0
prob += x32+x33-L<=0
#print(prob)
status = prob.solve()
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:6.0
    Point: {'L': 6.0, 'x11': 3.0, 'x12': 0.0, 'x13': 3.0, 'x22': 0.0, 'x23': 3.0, 'x24': 3.0, 'x32': 6.0, 'x33': 0.0}
    


```python
import cvxpy as cp
import numpy as np 

A1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 0, 1, 0, 0, 1, 0, 0], 
               [0, 0, 1, 0, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0]])
B1 = np.array([3, 6, 6, 3])
A2 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, -1],
               [0, 0, 0, 1, 1, 1, 0, 0, -1],
               [0, 0, 0, 0, 0, 0, 1, 1, -1]])
B2 = np.array([0, 0, 0])
C = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
n = 9
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(C @ x),
                [A1 @ x == B1, A2@x<=0, x>=0])
prob.solve()
print("cvxpy minimum value:", prob.value)
print("Point:", x.value)
```

    cvxpy minimum value: 5.999999999155014
    Point: [3.  1.5 1.5 1.5 1.5 3.  3.  3.  6. ]
    


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x1 = LpVariable('x1', 0, 1, cat='LpContinuous')
x2 = LpVariable('x2', 0, 1, cat='LpContinuous')
x3 = LpVariable('x3', 0, 1, cat='LpContinuous')
x4 = LpVariable('x4', 0, 1, cat='LpContinuous')
x5 = LpVariable('x5', 0, 1, cat='LpContinuous')
x6 = LpVariable('x6', 0, 1, cat='LpContinuous')
x7 = LpVariable('x7', 0, 1, cat='LpContinuous')
x8 = LpVariable('x8', 0, 1, cat='LpContinuous')
x9 = LpVariable('x9', 0, 1, cat='LpContinuous')
x10 = LpVariable('x10', 0, 1, cat='LpContinuous')
x11 = LpVariable('x11', 0, 1, cat='LpContinuous')
x12 = LpVariable('x12', 0, 1, cat='LpContinuous')
x13 = LpVariable('x13', 0, 1, cat='LpContinuous')

# 设置目标函数
prob += x1+x2+x3+x6+x8+x11+x12+x13+3.0000001*x4+3.0000001*x5+3.0000001*x9+3.0000001*x10+0.00000001*x7
# 约束条件


prob += x1+x4>=1
prob += x2+x4>=1
prob += x2+x5>=1
prob += x3+x5>=1
prob += x4+x6>=1
prob += x4+x7>=1
prob += x5+x7>=1
prob += x5+x8>=1
prob += x6+x9>=1
prob += x7+x9>=1
prob += x7+x10>=1
prob += x9+x11>=1
prob += x9+x12>=1
prob += x10+x12>=1
prob += x10+x13>=1

#print(prob)
status = prob.solve(use_mps=True)
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:8.00000001
    Point: {'x1': 1.0, 'x10': 0.0, 'x11': 1.0, 'x12': 1.0, 'x13': 1.0, 'x2': 1.0, 'x3': 1.0, 'x4': 0.0, 'x5': 0.0, 'x6': 1.0, 'x7': 1.0, 'x8': 1.0, 'x9': 0.0}
    


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x1 = LpVariable('x1', 0, 1, cat='LpContinuous')
x2 = LpVariable('x2', 0, 1, cat='LpContinuous')
x3 = LpVariable('x3', 0, 1, cat='LpContinuous')
x4 = LpVariable('x4', 0, 1, cat='LpContinuous')
x5 = LpVariable('x5', 0, 1, cat='LpContinuous')
x6 = LpVariable('x6', 0, 1, cat='LpContinuous')
x7 = LpVariable('x7', 0, 1, cat='LpContinuous')
x8 = LpVariable('x8', 0, 1, cat='LpContinuous')
x9 = LpVariable('x9', 0, 1, cat='LpContinuous')
x10 = LpVariable('x10', 0, 1, cat='LpContinuous')
x11 = LpVariable('x11', 0, 1, cat='LpContinuous')
x12 = LpVariable('x12', 0, 1, cat='LpContinuous')
x13 = LpVariable('x13', 0, 1, cat='LpContinuous')
x14 = LpVariable('x14', 0, 1, cat='LpContinuous')
x15 = LpVariable('x15', 0, 1, cat='LpContinuous')
x16 = LpVariable('x16', 0, 1, cat='LpContinuous')
x17 = LpVariable('x17', 0, 1, cat='LpContinuous')
x18 = LpVariable('x18', 0, 1, cat='LpContinuous')

# Target
prob += 9*x1+9*x5+9*x6+9*x16+9*x17+9*x18+30*x2+30*x3+30*x4+30*x7+30*x8+30*x9+30*x10+30*x11+30*x12+30*x13+30*x14+30*x15

# constraints
prob += x1+x2>=1
prob += x1+x5>=1
prob += x1+x6>=1
prob += x2+x3>=1
prob += x2+x4>=1
prob += x3+x4>=1
prob += x3+x5>=1
prob += x4+x6>=1
prob += x5+x6>=1
prob += x5+x17>=1
prob += x5+x16>=1
prob += x6+x10>=1
prob += x6+x17>=1
prob += x6+x18>=1
prob += x7+x13>=1
prob += x8+x13>=1
prob += x9+x11>=1
prob += x9+x12>=1
prob += x10+x14>=1
prob += x10+x15>=1
prob += x11+x16>=1
prob += x12+x17>=1
prob += x13+x17>=1
prob += x14+x17>=1
prob += x15+x18>=1

#print(prob)
status = prob.solve(use_mps=True)
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:175.5
    Point: {'x1': 0.5, 'x10': 1.0, 'x11': 0.0, 'x12': 0.0, 'x13': 1.0, 'x14': 0.0, 'x15': 0.0, 'x16': 1.0, 'x17': 1.0, 'x18': 1.0, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5, 'x5': 0.5, 'x6': 0.5, 'x7': 0.0, 'x8': 0.0, 'x9': 1.0}
    


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x1 = LpVariable('x1', 0, 1, cat='LpContinuous')
x2 = LpVariable('x2', 0, 1, cat='LpContinuous')
x3 = LpVariable('x3', 0, 1, cat='LpContinuous')
x4 = LpVariable('x4', 0, 1, cat='LpContinuous')
x5 = LpVariable('x5', 0, 1, cat='LpContinuous')
x6 = LpVariable('x6', 0, 1, cat='LpContinuous')
x7 = LpVariable('x7', 0, 1, cat='LpContinuous')
x8 = LpVariable('x8', 0, 1, cat='LpContinuous')
x9 = LpVariable('x9', 0, 1, cat='LpContinuous')
x10 = LpVariable('x10', 0, 1, cat='LpContinuous')
x11 = LpVariable('x11', 0, 1, cat='LpContinuous')
x12 = LpVariable('x12', 0, 1, cat='LpContinuous')
x13 = LpVariable('x13', 0, 1, cat='LpContinuous')

# Target
prob += 9*x1 + 9*x5 + 9*x6 + 20*x7 + 9*x8 + 9*x9 + 9*x13 + 30*x2 + 30*x3 + 30*x4 + 30*x12 + 30*x10 + 30*x11

# constraints
prob += x1+x2>=1
prob += x1+x5>=1
prob += x1+x6>=1
prob += x2+x3>=1
prob += x2+x4>=1
prob += x3+x4>=1
prob += x3+x5>=1
prob += x4+x6>=1
prob += x5+x6>=1
prob += x6+x7>=1
prob += x7+x8>=1
prob += x8+x10>=1
prob += x8+x9>=1
prob += x8+x13>=1
prob += x9+x11>=1
prob += x9+x13>=1
prob += x10+x11>=1
prob += x10+x12>=1
prob += x11+x12>=1
prob += x12+x13>=1

#print(prob)
status = prob.solve(use_mps=True)
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:126.0
    Point: {'x1': 0.5, 'x10': 0.5, 'x11': 0.5, 'x12': 0.5, 'x13': 0.5, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5, 'x5': 0.5, 'x6': 1.0, 'x7': 0.0, 'x8': 1.0, 'x9': 0.5}
    


```python
from pulp import *

prob = LpProblem('目标函数和约束', sense=LpMinimize)

x1 = LpVariable('x1', 0, 1, cat='LpContinuous')
x2 = LpVariable('x2', 0, 1, cat='LpContinuous')
x3 = LpVariable('x3', 0, 1, cat='LpContinuous')
x4 = LpVariable('x4', 0, 1, cat='LpContinuous')
x5 = LpVariable('x5', 0, 1, cat='LpContinuous')
x6 = LpVariable('x6', 0, 1, cat='LpContinuous')
x7 = LpVariable('x7', 0, 1, cat='LpContinuous')
x8 = LpVariable('x8', 0, 1, cat='LpContinuous')
x9 = LpVariable('x9', 0, 1, cat='LpContinuous')

prob += 9*x1 + 9*x2 + 9*x3 + 30*x4 + 30*x5 + 30*x6 + 100*x7 + 100*x8 + 100*x9

prob += x1+x2>=1
prob += x1+x3>=1
prob += x1+x4>=1
prob += x2+x3>=1
prob += x2+x5>=1
prob += x3+x6>=1
prob += x4+x7>=1
prob += x4+x5>=1
prob += x4+x6>=1
prob += x5+x8>=1
prob += x5+x6>=1
prob += x6+x9>=1
prob += x7+x8>=1
prob += x7+x9>=1
prob += x8+x9>=1

status = prob.solve(use_mps=True)
print(f"PuLp minimum value:{value(prob.objective)}")
print(f"Point:", {v.name: v.varValue for v in prob.variables()})
```

    PuLp minimum value:208.5
    Point: {'x1': 0.5, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5, 'x5': 0.5, 'x6': 0.5, 'x7': 0.5, 'x8': 0.5, 'x9': 0.5}
    


```python

```
