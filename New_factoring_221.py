from numpy import *
from sympy import *
import numpy as np

from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver, TabuSampler # classical free optimizer (installed locally on your computer)
from dwave.system import DWaveSampler, EmbeddingComposite # commercial quantum optimizer (in the cloud)

from dimod import BQM # binary quadratic model object

# Initialize pretty printing for symbolic mathematics
init_printing(use_unicode=True)
encoding = 'utf-8-sig'

# Define a constant value to be factorized (221)
print('Factorizing the number 221:')

# Define functions to handle different parts of the multiplication problem

# Calculate the product of elements in a block
def prodfunc(m):
    prodf = 0
    for r in range(2, (cols[m].shape[0] - 2)):
        for u in range(cols[m].shape[1]):
            prodf = prodf + cols[m][r, u] * 2 ** (cols[m].shape[1]-1-u)
    return prodf


# Calculate the carry from the m-1 block to the m block
def cinvalfunc(m):
    cinval = 0
    for r in range(cols[m].shape[1]):  # 第m块列数
        cinval = cinval + cols[m][(cols[m].shape[0]-2), r] * 2**(cols[m].shape[1]-1-r)
    return cinval


# Calculate the maximum carry from the m block to the m+1 block
def max_carryfunc(m):
    max_carry = 0
    if m < (len(cols)-1):
        for r in range(cols[m+1].shape[1]):
            max_carry = max_carry + cols[m+1][(cols[m].shape[0]-2), r] * 2**(cols[m+1].shape[1] + cols[m].shape[1]-1-r)
    return max_carry


# Calculate the target value for the m block
def targetvaluefunc(m):
    targetvalue = 0
    for r in range(cols[m].shape[1]):
        targetvalue = targetvalue + cols[m][cols[m].shape[0]-1, r] * 2**(cols[m].shape[1]-1-r)
    return targetvalue


# Define a function to handle 3-to-2 term conversion
def changefunc(bb, cc):
    tt = 0
    if p3*p1 == bb*cc:
        tt = t5
    if p3*p2 == bb*cc:
        tt = t6
    return tt


# Define a function to find matches in an expression
def findmatch(exprr, pattern):
    return [ee.match(pattern) for ee in exprr.find(pattern)]



# Main function
if __name__ == '__main__':
    # Define symbolic variables
    p1, p2, p3, p4, p5, p6, q1, q2, q3, q4, q5, q6, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, = symbols(
        'p1, p2, p3, p4, p5, p6, q1, q2, q3, q4, q5, q6, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11')
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18 = symbols(
        't1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18')
    t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36 = symbols(
        't19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36')

    x = [p3, q1, q2, c1, c2, c3, c4, t5, t6]
    p_list = [p3]
    q_list = [q1, q2]
    p_q_list = [p3, q1, q2]
    c_list = [c1, c2, c3, c4]
    t_list = [t5, t6]

    # Convert a decimal number to binary
    n = list(bin(221))
    binary_n = n[2:]  # Extract the binary representation of n

    # Convert binary string to a list of integers
    target_values = []
    for i in range(0, len(binary_n)):
        target_values = np.append(target_values, int(binary_n[i]))  # Convert each character to an integer

    # Define product terms for multiplication table
    p = [0, 0, 0, 1, p3, p2, p1, 1]
    q = [0, 0, 0, 0, 1, q2, q1, 1]
    product_terms1 = [0, 0, 0, 1, p3, p2, p1, 1]
    product_terms2 = [0, 0, q1, p3*q1, p2*q1, p1*q1, q1, 0]
    product_terms3 = [0, q2, p3*q2,	p2*q2, p1*q2, q2, 0, 0]
    product_terms4 = [1, p3, p2, p1, 1, 0, 0, 0]
    carries = [c4, c3, c2, c1, 0, 0, 0, 0]

    # Create a multiplication table as a numpy array
    A1 = np.vstack((p, q, product_terms1, product_terms2, product_terms3, product_terms4, carries, target_values))
    print(A1)

    # Split the multiplication table into blocks
    left0, cols0 = np.split(A1, [7], axis=1)  # Split the first block
    left1, cols1 = np.split(left0, [4], axis=1)  # Split the second block
    cols3, cols2 = np.split(left1, [2], axis=1)   # Split the third block
    cols = [cols0, cols1, cols2, cols3]

    # Define a cost function
    f1 = 0
    for i in range(1, len(cols)):
        f1 = f1 + (prodfunc(i) + cinvalfunc(i) - max_carryfunc(i) - targetvaluefunc(i)) ** 2

    # Simplify the function by replacing x^2 with x
    print('Original function:')
    f1 = expand(f1)
    for i in range(0, len(x)):
        f1 = f1.replace(x[i] ** 2, x[i])
    print(f1)

    # Modify the function based on a new algorithm
    f1 = f1.subs(p1, (1-q1))
    f1 = f1.subs(p2, (1-q2))
    f1 = expand(f1)
    for i in range(0, len(x)):
        f1 = f1.replace(x[i] ** 2, x[i])
        f1 = f1.replace(x[i] ** 3, x[i])
    print(f1)

    # Define Wild variables for pattern matching
    a = Wild("a", exclude=[Pow])
    b = Wild("b", exclude=[1, Pow])
    c = Wild("c", exclude=[1, Pow])
    d = Wild("d", exclude=[1, Pow])
    e = Wild("e", exclude=[1, Pow])
    y = Wild("y", exclude=[1, Pow])

    # First, eliminate the terms with a degree of four
    change_1 = findmatch(f1, a * b * c * d * e)  # Extract 3rd and 4th-degree terms, including the case when 'a' is 1
    ff = 0
    for i in range(len(change_1)):
        if change_1[i][b] in p_list:  # Check if it's a 4th-degree term
            f1 = f1 - change_1[i][a] * change_1[i][b] * change_1[i][c] * change_1[i][d] * change_1[i][e]
            if change_1[i][a] > 0:
                ff = ff + change_1[i][a]*(changefunc(change_1[i][b], change_1[i][d])*change_1[i][c]*change_1[i][e]+2*(change_1[i][b]*
                          change_1[i][d]-2*change_1[i][b]*changefunc(change_1[i][b], change_1[i][d])-2*change_1[i][d]*changefunc(change_1[i][b], change_1[i][d])+3*changefunc(change_1[i][b],
                                                                                                    change_1[i][d])))
            if change_1[i][a] < 0:
                ff = ff + change_1[i][a] * (changefunc(change_1[i][b], change_1[i][d]) * change_1[i][c] * change_1[i][e] - 2 * (change_1[i][b] *
                          change_1[i][d] - 2 *change_1[i][b] * changefunc(change_1[i][b], change_1[i][d]) - 2 *
                                                            change_1[i][d] * changefunc(change_1[i][b], change_1[i][d]) + 3 *changefunc(change_1[i][b], change_1[i][d])))
    f1 = f1 + ff
    # At this point, all 4th-degree terms have been converted to 3rd-degree terms
    # Now, convert 3rd-degree terms to 2nd-degree terms

    # Eliminate terms like cpq
    change_2 = findmatch(f1, a * b * c * d)
    ff = 0
    for i in range(len(change_2)):
        if change_2[i][b] in c_list:  # Check if it's a 3rd-degree term
            f1 = f1 - change_2[i][a] * change_2[i][b] * change_2[i][c] * change_2[i][d]
            if change_2[i][a] > 0:  
                ff = ff + change_2[i][a] * (changefunc(change_2[i][c], change_2[i][d]) * change_2[i][b] + 2 * (
                        change_2[i][c] * change_2[i][d] - 2 * change_2[i][c] * changefunc(change_2[i][c],
                                                                                          change_2[i][d]) -
                        2 * change_2[i][d] * changefunc(change_2[i][c], change_2[i][d]) + 3 * changefunc(
                         change_2[i][c], change_2[i][d])))
            elif change_2[i][a] < 0:  
                ff = ff + change_2[i][a] * (changefunc(change_2[i][c], change_2[i][d]) * change_2[i][b] - 2 * (
                        change_2[i][c] * change_2[i][d] - 2 * change_2[i][c] * changefunc(change_2[i][c],
                                                                                          change_2[i][d]) -
                        2 * change_2[i][d] * changefunc(change_2[i][c], change_2[i][d]) + 3 * changefunc(
                         change_2[i][c], change_2[i][d])))

    # Eliminate terms like ppq or pqq
    for i in range(len(change_2)):
        if change_2[i][b] in p_q_list and change_2[i][d] in q_list:
            f1 = f1 - change_2[i][a] * change_2[i][b] * change_2[i][c] * change_2[i][d]
            if change_2[i][c] in p_list:  # If the second unknown is 'p', combine 'b' and 'd'
                if change_2[i][a] > 0: # Check if the coefficient in front of the cubic term is positive
                    ff = ff + change_2[i][a] * (changefunc(change_2[i][b], change_2[i][d]) * change_2[i][c] + 2 * (
                            change_2[i][b] * change_2[i][d] - 2 * change_2[i][b] * changefunc(change_2[i][b],
                                                                                              change_2[i][d]) -
                            2 * change_2[i][d] * changefunc(change_2[i][b], change_2[i][d]) + 3 * changefunc(
                             change_2[i][b], change_2[i][d])))
                elif change_2[i][a] < 0: # Check if the coefficient in front of the cubic term is negative
                    ff = ff + change_2[i][a] * (changefunc(change_2[i][b], change_2[i][d]) * change_2[i][c] - 2 * (
                            change_2[i][b] * change_2[i][d] - 2 * change_2[i][b] * changefunc(change_2[i][b],
                                                                                              change_2[i][d]) -
                            2 * change_2[i][d] * changefunc(change_2[i][b], change_2[i][d]) + 3 * changefunc(
                             change_2[i][b], change_2[i][d])))
            if change_2[i][c] in q_list:  # If the second unknown is 'p', combine 'b' and 'd'
                if change_2[i][a] > 0: # Check if the coefficient in front of the cubic term is positive
                    ff = ff + change_2[i][a] * (changefunc(change_2[i][b], change_2[i][c]) * change_2[i][d] + 2 * (
                            change_2[i][b] * change_2[i][c] - 2 * change_2[i][b] * changefunc(change_2[i][b],
                                                                                              change_2[i][c]) -
                            2 * change_2[i][c] * changefunc(change_2[i][b], change_2[i][c]) + 3 * changefunc(
                             change_2[i][b], change_2[i][c])))
                elif change_2[i][a] < 0: # Check if the coefficient in front of the cubic term is negative
                    ff = ff + change_2[i][a] * (changefunc(change_2[i][b], change_2[i][d]) * change_2[i][c] - 2 * (
                            change_2[i][b] * change_2[i][c] - 2 * change_2[i][b] * changefunc(change_2[i][b],
                                                                                              change_2[i][c]) -
                            2 * change_2[i][c] * changefunc(change_2[i][b], change_2[i][c]) + 3 * changefunc(
                             change_2[i][b], change_2[i][c])))

    # 消掉三项为pqt的
    for i in range(len(change_2)):
        if change_2[i][b] in p_list and change_2[i][d] in t_list:  # 判断是否为三次项pqt
            f1 = f1 - change_2[i][a] * change_2[i][b] * change_2[i][c] * change_2[i][d]
            if change_2[i][a] > 0:  # 判断三次项前面常数正负
                ff = ff + change_2[i][a] * (changefunc(change_2[i][b], change_2[i][c]) * change_2[i][d] + 2 * (
                            change_2[i][b] * change_2[i][c] - 2 * change_2[i][b] * changefunc(change_2[i][b],
                                                                                              change_2[i][c]) -
                            2 * change_2[i][c] * changefunc(change_2[i][b], change_2[i][c]) + 3 * changefunc(
                             change_2[i][b], change_2[i][c])))
            elif change_2[i][a] < 0:  # 判断三次项前面常数正负
                ff = ff + change_2[i][a] * (changefunc(change_2[i][b], change_2[i][c]) * change_2[i][d] - 2 * (
                        change_2[i][b] * change_2[i][c] - 2 * change_2[i][b] * changefunc(change_2[i][b],
                                                                                          change_2[i][c]) -
                        2 * change_2[i][c] * changefunc(change_2[i][b], change_2[i][c]) + 3 * changefunc(
                         change_2[i][b], change_2[i][c])))

    f1 = f1 + ff    # Add the resulting quadratic terms
    print('Function after converting to quadratic terms:')
    print(f1)   # Up to this point, everything has been converted to quadratic terms

    # Convert to particle spin form
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12 = symbols(
        's1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12')  # 定义二进制未知变量
    s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24 = symbols(
        's13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24')  # 定义二进制未知变量
    s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36 = symbols(
        's25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36')  # 定义二进制未知变量
    s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48 = symbols(
        's37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48')  # 定义二进制未知变量
    s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59 = symbols(
        's49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59')  # 定义二进制未知变量
    s = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    for i in range(len(x)):
        f1 = f1.subs(x[i], ((1 - s[i]) / 2))
    f1 = 2*f1
    f1 = expand(f1)
    print("ISing 模型多项式：")
    print(f1)

    # Extract h and J
    dic = findmatch(f1, a * b * c)
    h_list = np.zeros(len(x))
    J_list = np.zeros((len(x), len(x)))
    h = {}
    J = {}
    for i in range(len(dic)):
        if dic[i][b] in s:
            if s.index(dic[i][b]) < s.index(dic[i][c]):
                J_list[s.index(dic[i][b])][s.index(dic[i][c])] = dic[i][a]
            else:
                J_list[s.index(dic[i][c])][s.index(dic[i][b])] = dic[i][a]
        else:
            h_list[s.index(dic[i][c])] = dic[i][b]

    for i in range(len(h_list)):
        h.setdefault(i, h_list[i])

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            J.setdefault((i, j), J_list[i][j])
    print('Extracted h:')
    print(h)
    print('Extracted J:')
    print(J) 
    print()

    # Call the QBSolv third-party library
    count_all_first = 1  # Number of times the third-party library is called
    correct_count = 0
    count = 0
    q_truth_1 = 13  # Input prime factor
    q_truth_2 = 17  # Input prime factor
    for i in range(count_all_first):
    
        bqm = BQM.from_ising(h, J)
        optimizer = SimulatedAnnealingSampler()
        sampleset = optimizer.sample(bqm=bqm, num_reads=1000)
        
        print("Particle Spin State:")
        spin = list(sampleset.samples())
        print(spin)
        
        print("Energy Values Corresponding to Particle Spin States:")
        energy = list(sampleset.data_vectors['energy'])
        print(energy)

        # Reverse-engineer the value of q
        q = 2 ** (len(q_list) + 1) + 1
        for j in range(len(q_list)):
            q_list[j] = (1 - spin[0][len(p_list) + j]) / 2
        for j in range(len(q_list)):
            q = q + q_list[j] * 2 ** (j + 1)
        
        print('Value of q:')
        print(q)
        
        if q == q_truth_1 or q == q_truth_2:
            correct_count = correct_count + 1
        else:
            # If the energy value is smaller than the first one, then# If the energy value is smaller than the first one, then
            for k in range(1, len(energy)):
                if energy[k] <= energy[0]:
                    count = count + 1
                    q = 2 ** (len(q_list) + 1) + 1
                    for j in range(len(q_list)):  # Find the corresponding q value
                        q_list[j] = (1 - spin[k][len(p_list) + j]) / 2
                    for j in range(len(q_list)):
                        q = q + q_list[j] * 2 ** (j + 1)
                    if q == q_truth_1 or q == q_truth_2:
                        correct_count = correct_count + 1
                    print('Alternatively, it could be: ', end='')
                    print(q)
                    
    count_all = count_all_first + count
    print('Minimum value of h: ', end='')
    print(min(h.values()), end='')
    print(', Maximum value of h: ', end='')
    print(max(h.values()))
    print('Minimum value of J: ', end='')
    print(min(J.values()), end='')
    print(', Maximum value of J: ', end='')
    print(max(J.values()))
    print('Total times lowest energy was obtained: ', end='')
    print(count_all)
    print('Number of correct times: ', end='')
    print(correct_count)
    print('Accuracy: ', end='')
    print(correct_count / count_all)
    print('Number of bits used: ', end='')
    print(len(s))
