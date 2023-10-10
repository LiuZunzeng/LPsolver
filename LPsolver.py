# -*- coding:utf-8 -*-
"""
@Time:2023/10/3
@Auth:Liu Zunzeng
@File:LPsolver.py
"""
import numpy as np

class Model:

    def __init__(self):

        self.solutionStatus = None
        self.basicVars = []
        self.artificialVars = []
        self.nonbasicVars = []

    def build(self, valueVector, coefficientMatrix, resourceVector):
        """
        Input the optimization model: max or min z = cx subject to Ax = b, x>=0
        """
        self.c = np.array(valueVector).astype(float)
        self.A = np.array(coefficientMatrix).astype(float)
        self.b = np.array(resourceVector).astype(float)
        self.initial_A = np.array(coefficientMatrix).astype(float)

    def optimize(self, direction, method = 'simplex'):
        """
        Call simplex method or interior point method to solve the model
        """
        row_num = len(self.A)
        col_num = len(self.A[0])

        # Construct an m×m identity matrix
        unitMatrix = np.identity(row_num)

        col_index = 0

        for j in range(col_num):

            is_unitVector = True

            # Determine whether the vector in column j is a unit vector
            for i in range(row_num):
                if (self.A[i][j] != unitMatrix[i][col_index]):
                    is_unitVector = False

            # if vector j is a unit vector, then x_j becomes basic variable
            if (is_unitVector == True):
                col_index += 1
                self.basicVars.append(j)

        # check if we find a complete unit matrix in coefficient matrix A, if not, we need to add artificial variables
        if (col_index + 1 < row_num):

            incompleteMatrix = unitMatrix[:, col_index: row_num]
            self.A = np.append(self.A, incompleteMatrix, axis=1)
            self.initial_A = self.A

            M = 1e+8

            for i in range(row_num - col_index):
                self.artificialVars.append(col_num + i)
                self.c = np.append(self.c, M)

            # artificial variables become basic variables
            self.basicVars = self.basicVars + self.artificialVars

        col_num = len(self.A[0])

        # Vars(list) is the union of basicVars(list) and nonbasicVars(list)
        temp = self.basicVars
        Vars = []
        for i in range(col_num):
            Vars.append(i)
        temp.reverse()
        for i in temp:
            Vars.pop(i)
        self.nonbasicVars = Vars
        self.basicVars.sort()

        # decide which algorithm to use
        if(method == 'simplex'):
            self.simplex(direction)
        if(method == 'interior'):
            self.interior(direction)

    def simplex(self, direction):
        """
        Use simplex method to solve the LP
        """

        # because of adding artificial variables, there may be more columns.
        print("Algorithm used: Simplex method")

        row_num = len(self.A)
        col_num = len(self.A[0])

        # c_B is the value coefficient of basic variables while C_N is of nonbasic variables
        c_B = np.zeros(shape=(row_num,), dtype=float)
        c_N = np.zeros(shape=(col_num - row_num,), dtype=float)
        for i in range(row_num):
            c_B[i] = self.c[self.basicVars[i]]
        for i in range(col_num - row_num):
            c_N[i] = self.c[self.nonbasicVars[i]]

        # we also need to extract the coefficient matrix of nonbasic variables for calculation of test numbers
        A_N = np.zeros(shape=(row_num, col_num - row_num), dtype=float)
        for i in range(row_num):
            for j in range(col_num - row_num):
                A_N[i][j] = self.A[i][self.nonbasicVars[j]]

        B = np.zeros(shape=(row_num, row_num), dtype=float)
        for i in range(row_num):
            for j in range(row_num):
                B[i][j] = self.initial_A[i][self.basicVars[j]]
        p = np.dot(np.linalg.inv(B.T), c_B)

        # calculate the test numbers
        reducedCost = c_N - np.dot(c_B, A_N)

        print("----------------------Initial coefficient matrix and vectors-----------------------")
        print("A:", self.A)
        print("b:", self.b)
        print("dual vars:", p)
        print("reducedCost:", reducedCost)

        eps = -1e-6

        if (direction == 'min'):
            extr_sigma = min(reducedCost)
            is_loop = extr_sigma <= -eps
        elif (direction == 'max'):
            extr_sigma = max(reducedCost)
            is_loop = extr_sigma >= eps

        iterNum = 1

        while (is_loop):

            print("-----------------------------The %dth iteration begins------------------------------" % iterNum)

            # determine which nonbasic variable enter basis
            if (direction == 'min'):
                enterVar = self.nonbasicVars[np.argmin(reducedCost)]
            elif (direction == 'max'):
                enterVar = self.nonbasicVars[np.argmax(reducedCost)]

            print("enterVar_index:", enterVar)

            leaveVar_index = 0
            min_ratio = 999999

            # determine which basic variable leave basis according to the minimum ratio principle
            for row_index in range(row_num):
                if (self.A[row_index][enterVar] != 0):
                    ratio = self.b[row_index] / self.A[row_index][enterVar]
                    if (ratio > 0 and ratio < min_ratio):
                        min_ratio = ratio
                        leaveVar_index = row_index
            print("leaveVar_index:", self.basicVars[leaveVar_index])

            # if we can not find a ratio greater than 0, the LP's best solution is unbounded
            if (min_ratio == 999999):
                self.solutionStatus = "Unbounded solution"
                break

            # Gaussian elimination
            # update pivot row
            leaveVar = self.basicVars[leaveVar_index]
            self.basicVars[leaveVar_index] = enterVar
            self.nonbasicVars.remove(enterVar)
            self.nonbasicVars.append(leaveVar)
            pivot_num = self.A[leaveVar_index][enterVar]

            for col in range(col_num):
                self.A[leaveVar_index][col] = self.A[leaveVar_index][col] / pivot_num
            self.b[leaveVar_index] = self.b[leaveVar_index] / pivot_num

            # update other rows
            for row in range(row_num):

                if (row != leaveVar_index):
                    step_num = self.A[row][enterVar]

                    for col in range(col_num):
                        self.A[row][col] = self.A[row][col] - self.A[leaveVar_index][col] * step_num
                    self.b[row] = self.b[row] - self.b[leaveVar_index] * step_num

            print("new A:", self.A)
            print("new b:", self.b)

            # update c_B, c_N and A_N
            for i in range(len(self.nonbasicVars)):
                c_N[i] = self.c[self.nonbasicVars[i]]
            for i in range(len(self.basicVars)):
                c_B[i] = self.c[self.basicVars[i]]
            for i in range(row_num):
                for j in range(len(self.nonbasicVars)):
                    A_N[i][j] = self.A[i][self.nonbasicVars[j]]

            # calculate the test numbers to check if the basic variables constitute the best basis
            reducedCost = c_N - np.dot(c_B, A_N)

            B = np.zeros(shape=(row_num, row_num), dtype=float)
            for i in range(row_num):
                for j in range(row_num):
                    B[i][j] = self.initial_A[i][self.basicVars[j]]
            p = np.dot(np.linalg.inv(B.T), c_B)
            print("dual vars:", p)

            print("reducedCost:", reducedCost)

            A_N = np.zeros(shape=(row_num, col_num - row_num), dtype=float)
            for i in range(row_num):
                for j in range(col_num - row_num):
                    A_N[i][j] = self.A[i][self.nonbasicVars[j]]


            # The loop termination condition is determined according to the orientation of the objective function
            if (direction == "min"):
                extr_sigma = min(reducedCost)
                is_loop = extr_sigma <= -eps
            elif (direction == "max"):
                extr_sigma = max(reducedCost)
                is_loop = extr_sigma >= eps

            iterNum += 1

        print("------------------------------Iteration completion---------------------------------")
        print("A total of", iterNum - 1, "iterations")
        if (self.artificialVars is not None):
            for artificialVar in self.artificialVars:
                if (artificialVar in self.basicVars):
                    self.solutionStatus = "Infeasible solution"

        if (self.solutionStatus != "Unbounded solution" and self.solutionStatus != "Infeasible solution"):
            for i in range(len(reducedCost)):
                if (reducedCost[i] == 0):
                    self.solutionStatus = "Alternative optimal solution"
                    break
                else:
                    self.solutionStatus = "Optimal"

            solution = [0 for i in range(col_num)]
            for i in range(row_num):
                solution[self.basicVars[i]] = self.b[i]

            z_opt = np.dot(c_B, self.b)

            print("solution_status:", self.solutionStatus)
            print("objective value:", z_opt)
            print("optimal solution:", solution)

        else:
            print("solution_status:", self.solutionStatus)

    def interior(self, direction):
        '''
        Affine scaling algorithm
        '''

        print("Algorithm used: affine scaling algorithm")

        if(direction == "max"):
            for i in range(len(self.c)):
                self.c[i] = - self.c[i]

        row_num = len(self.A)
        col_num = len(self.A[0])

        solution = [0 for i in range(col_num)]
        #for i in range(row_num):
            #solution[self.basicVars[i]] = self.b[i]
        for nonbasicVar in self.nonbasicVars:

            solution[nonbasicVar] = 0.2

        for i in range(row_num):

            sum = 0

            for j in self.nonbasicVars:

                sum += solution[j] * self.A[i][j]

            solution[self.basicVars[i]] = self.b[i] - sum

        #solution = [0.1, 0.1, 1.8, 1]
        print(" initial solution:", solution)

        X = np.diagflat([solution])
        print("X:", X)
        inverseMatrix = np.linalg.inv(np.dot(np.dot(self.A, np.dot(X, X)), self.A.T))
        print("inverseMatrix:", inverseMatrix)
        print("A", self.A)
        print("c", self.c)
        p = np.dot(np.dot(np.dot(inverseMatrix, self.A), np.dot(X, X)), self.c)
        print("p:",p)
        r = self.c - np.dot(self.A.T, p)
        print("r:", r)
        e = np.array([1 for i in range(col_num)])
        gap = np.dot(np.dot(e, X), r)
        print("initial gap:", gap)
        d = np.dot(np.dot(-X, X), r)
        print("d:", d)
        eps = 1e-3
        tolerance = -1e-6
        beta = 0.995
        iterNum = 1



        while(gap >= eps or not (r >= tolerance).all()):

            print("-----------------------------The %dth iteration begins------------------------------" % iterNum)

            d = np.dot(np.dot(-X, X), r)
            print("direction vector:", d)
            if((d >= 0).all()):
                self.solutionStatus = "Unbounded solution"
                break
            else:
                step = beta/np.linalg.norm(d)
                ratio = [ 99999 for i in range(col_num)]

                print("step size:", step)

                for i in range(len(solution)):
                    if(d[i] < 0):
                        ratio[i] = -solution[i] / d[i]

                if(step > min(ratio)):
                    step = min(ratio)

                for i in range(len(solution)):
                    solution[i] = solution[i] + step * d[i]

                print("new solution:", solution)
                X = np.diagflat([solution])
                p = np.dot(
                    np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.A, np.dot(X, X)), self.A.T)), self.A), X), X),
                    self.c)
                print("c:", self.c)
                print("A'", self.A.T)
                print("p:",p)
                r = self.c - np.dot(self.A.T, p)
                print("r:",r)
                e = np.array([1 for i in range(col_num)])
                gap = np.dot(np.dot(e, X), r)
                print("gap:", gap)
                if (direction == "min"):
                    z_opt = np.dot(self.c, solution)
                if (direction == "max"):
                    z_opt = -np.dot(self.c, solution)
                print("z_opt:",z_opt)

            iterNum += 1

        print("------------------------------Iteration completion---------------------------------")
        print("A total of", iterNum - 1, "iterations")

        if (self.artificialVars is not None):
            for artificialVar in self.artificialVars:
                if(solution[artificialVar] > 0):
                    self.solutionStatus = "No feasible solution found"

        if(self.solutionStatus != "Unbounded solution" and self.solutionStatus != "No feasible solution found"):
            self.solutionStatus = "%f-optimal"%eps
            if(direction == "min"):
                z_opt = np.dot(self.c, solution)
            if(direction == "max"):
                z_opt = -np.dot(self.c, solution)
            print("solution_status:", self.solutionStatus)
            print("objective value:", z_opt)
            print("optimal solution:", solution)

        else:
            print("solution_status:", self.solutionStatus)

if __name__ == '__main__':
    '''
            min z = -3 * x_1 + x_2 + x_3
    subject to:
                 x_1 - 2 * x_2 +     x_3 <= 11
            -4 * x_1 +     x_2 + 2 * x_3 >= 3
            -2 * x_1           +     x_3 = 1

    standard form:
            min z = -3 * x_1 + x_2 + x_3 + 0 * x_4 + 0 * x_5 
    subject to:
                 x_1 - 2 * x_2 +     x_3 + x_4       = 11
            -4 * x_1 +     x_2 + 2 * x_3       - x_5 = 3
            -2 * x_1           +     x_3             = 1        
    '''
    '''

    m = Model()
    m.build(valueVector = [-3, 1, 1, 0, 0],
            coefficientMatrix = [[1, -2, 1, 1, 0]
                         , [-4, 1, 2, 0, -1]
                         , [-2, 0, 1, 0, 0]],
            resourceVector = [11, 3, 1])

    m.optimize("min", method="simplex")
    '''
    '''
            max z = 2 * x_1 + 3 * x_2 
    subject to:
                 x_1 + 2 * x_2 <= 8
             4 * x_1           <= 16
                       4 * x_2 <= 12   
                      x_1, x_2 >= 0

    standard form:
            max z = 2 * x_1 + 3 * x_2 + 0 * x_3 + 0 * x_4 + 0 * x_5
    subject to:
                 x_1 + 2 * x_2 + x_3             = 8
             4 * x_1                 + x_4       = 16
                       4 * x_2             + x_5 = 12   
                         x_1, x_2, x_3, x_4, x_5 >= 0      
    '''
    #'''
    m = Model()
    m.build(valueVector=[2, 3, 0, 0, 0],
            coefficientMatrix=[[1, 2, 1, 0, 0]
             , [4, 0, 0, 1, 0]
             , [0, 4, 0, 0, 1]],
            resourceVector=[8, 16, 12])
    m.optimize("max", method="interior")
    #'''
    '''
                max z = x_1 + 2 * x_2 
        subject to:
                     x_1 + x_2 <= 2
                   - x_1 + x_2 >= 3

        standard form:
                min z = - x_1 - 2 * x_2
        subject to:
                    x_1 + x_2 + x_3 = 2
                  - x_1 + x_2       + x_4 = 3
    '''
    '''
    m = Model()
    m.build(valueVector=[1, 2, 0, 0],
            coefficientMatrix=[[1, 1, 1, 0]
             , [-1, 1, 0, 1]],
            resourceVector=[2, 1])
    m.optimize("max", method="simplex")
    '''