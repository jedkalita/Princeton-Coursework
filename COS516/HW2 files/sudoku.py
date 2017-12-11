from z3 import *
import sys

# A variable representing the value of a specific cell
def matrixvar(i, j):
    return Int("x_%s_%s" % (i,j))

# Create a 9x9 matrix of integer variables
def getMatrix():
    return [ [ matrixvar(i+1, j+1) for j in range(9) ] 
             for i in range(9) ]


# Students should add their code in the following 4 functions
# (instead of the 'return True' statement)

# Add constraints such that each cell contains a value in {1, ..., 9} 
def addCellConstraints(X):
    each_ele = []
    for i in range(9):
        for j in range(9):
            each_ele.append(And(X[i][j] >= 1, X[i][j] <= 9))
            
    #each_ele = [ [ And(X[i][j] >= 1, X[i][j] <= 9) for j in range(9) ]
            #     for i in range(9)] 
    #print(And(each_ele))
    return(And(each_ele))
    #return True

# Add constraints such that each row contains a digit at most once
def addRowConstraints(X):
    each_row = []
    count = 0
    for i in range(9):
        for j in range(9):
            for k in range(1,10):
                each_row.append(Implies(X[i][j] == k, And([Or(X[i][r] < k, X[i][r] > k) for r in range(9) if r is not j])))
                #count = count + 1

    #print('\nNOW printing addRowConstraints()\n')
    #print(count)
    #print(each_row)
    return(And(each_row))
    #return True

# Add constraints such that each column contains a digit at most once
def addColumnConstraints(X):
    each_col = []
    count = 0
    for i in range(9):
        for j in range(9):
            for k in range(1,10):
                each_col.append(Implies(X[j][i] == k, And([Or(X[r][i] < k, X[r][i] > k) for r in range(9) if r is not j])))
                count = count + 1                                               

    #print('\nNOW printing addRowConstraints()\n')                               
    #print(count)                                                                
    #print(each_col)                                                             
    return(And(each_col))
    #return True

# Add constraints such that each 3x3 square contains a digit at most once
def addSquareConstraints(X):
    square_1 = []
    square_2 = []
    square_3 = []
    square_4 = []
    square_5 = []
    square_6 = []
    square_7 = []
    square_8 = []
    square_9 = []
    for i in range(0,3):
        for j in range(0,3):
            for k in range(1,10):
                square_1.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(0,3) for t in range(0,3) if any([r is not i, t is not j])])))
    
    #print('Square 1.')
    #print(square_1)
                                                      
    for i in range(0,3):
        for j in range(3,6):
            for k in range(1,10):
                square_2.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(0,3) for t in range(3,6) if any([r is not i, t is not j])])))
 

    for i in range(0,3):
        for j in range(6,9):
            for k in range(1,10):
                square_3.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(0,3) for t in range(6,9) if any([r is not i, t is not j])])))

    for i in range(3,6):
        for j in range(0,3):
            for k in range(1,10):
                square_4.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][
t] > k) for r in range(3,6) for t in range(0,3) if any([r is not i, t is not j])])))

    for i in range(3,6):
        for j in range(3,6):
            for k in range(1,10):
                square_5.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(3,6) for t in range(3,6) if any([r is not i, t is not j])])))

    for i in range(3,6):
        for j in range(6,9):
            for k in range(1,10):
                square_6.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(3,6) for t in range(6,9) if any([r is not i, t is not j])
])))
    
    for i in range(6,9):
        for j in range(0,3):
            for k in range(1,10):
                square_7.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(6,9) for t in range(0,3) if any([r is not i, t is not j])])))

    for i in range(6,9):
        for j in range(3,6):
            for k in range(1,10):
                square_8.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(6,9) for t in range(3,6) if any([r is not i, t is not j])])))


    for i in range(6,9):
        for j in range(6,9):
            for k in range(1,10):
                square_9.append(Implies(X[i][j] == k, And([Or(X[r][t] < k, X[r][t] > k) for r in range(6,9) for t in range(6,9) if any([r is not i, t is not j])])))

    sq_1 = And(square_1)
    sq_2 = And(square_2)
    sq_3 = And(square_3)
    sq_4 = And(square_4)
    sq_5 = And(square_5)
    sq_6 = And(square_6)
    sq_7 = And(square_7)
    sq_8 = And(square_8)
    sq_9 = And(square_9)
    sq = And(sq_1, sq_2, sq_3, sq_4, sq_5, sq_6, sq_7, sq_8, sq_9)
    #print(sq)

    #for i in range(0:3:9):
    #    for j in range(3):
    #        for k in range(1,10):
    #            square.append(Implies(X[i][j] == k, And([Or(X[r][j] < k, X[r][j] > k) for r in range(
    #    for j in range(3,6):
            
    #    for j in range(6,9):
    return sq
    #return True

def solveSudoku(instance):
    X = getMatrix()
    #print(type(X[1][1]))
    #print(type(instance[1][1]))
    #print('\n Printing contents of the matrix instance:\n')
    #print(instance)     
    # Create the initial constraints of the puzzle
    # based on the input instance. Note that '0' represents 
    # an empty cells
    instance_c = [ If(instance[i][j] == 0, 
                      True, 
                      X[i][j] == instance[i][j]) 
                   for i in range(9) for j in range(9) ]
    #print(instance_c)
    #print(Int(X[8][7]))
    #print("Printing the matrix instance: ", [ [ matrixvar(i+1, j+1) for j in range(9) ] for i in range(9) ]   )
    # Create the Z3 solver
    s = Solver()

    # Add all needed constraints
    s.add(instance_c)
    s.add(addCellConstraints(X))
    s.add(addRowConstraints(X))
    s.add(addColumnConstraints(X))
    s.add(addSquareConstraints(X))

    # If the problem is satisfiable, a solution exists
    if s.check() == sat:
        m = s.model()
        r = [ [ m.evaluate(X[i][j]) for j in range(9) ] 
              for i in range(9) ]
        print_matrix(r)
    else:
        print "failed to solve"
