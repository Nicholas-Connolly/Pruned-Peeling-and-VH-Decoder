import numpy as np
import random
from itertools import chain, combinations
#import matplotlib.pyplot as plt

from Hypergraph_Product_Code_Construction_v3 import HGP_code
from Hypergraph_Product_Code_Construction_v3 import standard_form


# Functions to convert between the 1-dimensional and 2-dimensional indexing used in the HGP code construction.
# Used to recover the array structure of HGP code indices.
# Begins indexing at 0.
# The number of rows is actually not needed?

# The HGP code construction can be viewed geometrically in the following way.
# Given two Tanner graphs G1 = (B1,C1) and G2 = (B2,C2), where B1, B2 represent bits and C1, C2 represent checks,
# these can be combined into the Tanner graph G = (B1xB2 \cup C1xC2, B1xC2 \cup C1xB2), with the following visualization:
#
#            B2       C2
#   B1  [  B1xB2  |  B1xC2  ]
#   C1  [  C1xB2  |  C1xC2  ]
#
# To convert between indices and biindices, we use a combination of floor division and modular division.
# The wrapping is determined by the number of columns in each of these two blocks, which come from the sizes of B2 and C2.
# Hence, the number of rows is not needed for this conversion
#
# We must also account for an index_shift case of working with "vertical qubits" from the block C1xC2.
# The vertical qubits are generally indexed after the horizontal qubits, but they have different block structures.
# In this case, the value of index_shift is the number of horizontal qubits.
# By default, this value is set to 0.

def index_to_biindex(index,num_cols,index_shift=0):
    return (index-index_shift)//num_cols, (index-index_shift)%num_cols

def biindex_to_index(biindex,num_cols,index_shift=0):
    return biindex[0]*num_cols + biindex[1] + index_shift



# Function to generate a random erasure pattern as a binary vector based on a given rate.
# An entry is flipped from 0 to 1 based on the probability specified.

def generate_erasure_pattern_vector(n,erasure_rate):
    
    erasure_vect = np.zeros(n,dtype=int)
    
    for i in range(n):
        if (random.random() <= erasure_rate):
            erasure_vect[i] = 1
            
    return erasure_vect


# function to generate a random error vector supported on this erasure.

def generate_random_error_vector_with_erasure_support(erasure_vect,error_rate):
    
    error_vect = np.zeros(len(erasure_vect),dtype=int)
    
    for i in range(len(erasure_vect)):
        if (erasure_vect[i]==1):
            if (random.random() <= error_rate):
                error_vect[i]=1
                
    return error_vect


# Function combining the erasure and error generation functions.

def generate_random_erasure_and_error(n,erasure_rate,error_rate):

    erasure_vect = generate_erasure_pattern_vector(n,erasure_rate)
    error_vect = generate_random_error_vector_with_erasure_support(erasure_vect,error_rate)

    return erasure_vect, error_vect



### FUNCTIONS TO GENERATRE RANDOM ERASURE AND ERRORS USING SETS OF INDICES FOR NON-ZERO ENTRIES

# Function to generate a random erasure pattern as a set of indices for non-zero entries in some vector
# An entry is flipped from 0 to 1 based on the probability specified.

def generate_erasure_pattern_index_set(n,erasure_rate):
    
    erasure_index_set = set()
    
    for i in range(n):
        if (random.random() <= erasure_rate):
            erasure_index_set.add(i)
            
    return erasure_index_set


# function to generate a random set of indices of erroneous qubits based on a given error rate.
# This set of indices is a subset of a given set of erased qubit indices.

def generate_random_error_index_set_with_erasure_support(n,erasure_index_set,error_rate):
    
    error_index_set = set()
    
    for i in range(n):
        if (i in erasure_index_set):
            if (random.random() <= error_rate):
                error_index_set.add(i)
                
    return error_index_set


# Function combining the erasure and error generation functions using index sets of nonzero entries (rather than binary vectors).
# Must specify a number of qubits n, and the erasure_rate and error_rate.

def generate_random_erasure_and_error_index_sets(n,erasure_rate,error_rate):

    erasure_index_set = generate_erasure_pattern_index_set(n,erasure_rate)
    error_index_set = generate_random_error_index_set_with_erasure_support(n,erasure_index_set,error_rate)

    return erasure_index_set, error_index_set



# Function to "zero" any columns of a parity check matrix corresponding to non-erased qubits.
# Used as a way to examine the modified parity check matrix of a code involving an erasure.
# In this way, qubits not in the erasure are ignored.

# Function inputs:
# H: a parity check matrix.
# erasure: a binary vector denoting erased qubits (length much match the number of columns in H)

def erase_qubit_cols_from_parity_check_matrix(H,erasure):
    
    H_prime = np.copy(H)
    num_rows = np.shape(H)[0]
    num_cols = np.shape(H)[1]
    
    if (len(erasure) != num_cols):
        raise Exception("The number of erased qubits does not match the number of columns in H.")
    else:
        for j in range(len(erasure)):
            if (erasure[j] == 0):
                for i in range(num_rows):
                    H_prime[i][j]=0
                    
    return H_prime



# Function to construct a list of sets of qubit indices matching each adjacent check index.
# This is structured as a list of sets, obtained from the parity check matrix:
#     The index in the outer-list denotes the check, this index matches the row within the parity check matrix.
#     The inner-set consists of a set of qubit indices, each matching their column within the parity check matrix.
# This same function can also be used to construct a lsit of sets of qubit indices matching each adjacent generator index.
# Similarly, using the transpose, a list of check/generator adjacent to each qubit can also be obtained.

# This is essentially an adjaceny list for a bipartite graph.
# Thus function should be included as part of the utilties?
# These lists should be part of the HGP class object?

# Function Inputs:
# H: a binary matrix (such as a parity check matrix)
# col_index_shift: an integer used to shift the index of each column; (used in the special case of vertical qubits).

# Function Outputs
# Adajency_list: a list of sets of column indices denoting those columns which have a non-zero entry in this row.

def compute_adjacency_list(H,col_index_shift=0):
    
    num_rows, num_cols = np.shape(H)
    adjacency_list = []
    
    for i in range(num_rows):
        temp_row_set = set()
        for j in range(num_cols):
            # Add the column index to this list if the corresponding entry in H is nonzero (this allows for multi-edges)
            if(H[i][j] != 0):
                temp_row_set.add(j+col_index_shift)
        adjacency_list.append(temp_row_set)
        
    return adjacency_list



# Function to convert an adjaceny list for a Tanner graph to a binary matrix.
# Used because the default version of the HGP construction uses binary matrices.

# Function Inputs:
# num_rows: the number of rows in the adjacency matrix = number of checks in the Tanner graph.
# num_cols: then number of columns in the adjaceny matrix = number of bits in the Tanner graph.
# adj_list: a 2-dimensional numpy array (or list of lists) representing the adjaceny list for this graph.

# Function Outputs:
# H: the binary parity check matrix obtained from this adjacency list.

def convert_adjacency_list_to_binary_matrix(num_rows,num_cols,adj_list):
    
    # Initialize a matrix of zeros of the appropriate size.
    H = np.zeros((num_rows,num_cols),dtype=int)
    
    for row_index in range(num_rows):
        # Fix a row from the adjacency list
        local_row = adj_list[row_index]
        
        # Iterate through the column indices in this row and set each of them equal to 1.
        for col_index in local_row:
            H[row_index,col_index] = 1
            
    # Return the binary matrix
    return H



# Function to write to a text file the adjacency list from a given binary matrix.
# The values in this adjacency list are separate by spaces, and each row is separated by a new line break.
# The first line in the file is a a pair of numbers "n m" which are NOT part of the adjacency list.
# Rather, n denotes the number of bits and m denotes the number of checks (needed for conversion elsewhere)
# Unless a file name is specified, a default file name is generated based on the parameters of the binary matrix.

# Function Inputs:
# H_input: the binary matrix to convert (generally a parity check matrix)
# file_name: the file name that will be used for the output text file; a default name is generated if not specified.

# Function Outputs:
# saves a .txt in the working directory where this function is called.

def write_adj_list_to_file_from_H(H_input,file_name="default"):
    
    # Infer the number of checks and bits based on this parity check matrix.
    num_checks, num_bits = H_input.shape
    
    # Infer the degrees of each bit and check, assuming constant weight rows and columns.
    bit_deg = np.count_nonzero(H_input[0])
    check_deg = np.count_nonzero(H_input[:,0])
    
    if (file_name == "default"):
        # The default file name is based on the number of bits and the bit degree, and the number of checks and check degree.
        # Example: "adj_list_8bd4_6cd3" means 8 bits of degree 4, 6 checks of degree 3.
        file_name = "adj_list_"+str(num_bits)+"bd"+str(bit_deg)+"_"+str(num_checks)+"cd"+str(check_deg)
        
    # Add the file extension to the name as well.
    file_name = file_name+str(".txt")
    
    # Convert this parity check matrix into an adjacency list.
    adj_list = compute_adjacency_list(H_input)
    
    with open(file_name, 'w') as file:
        file.write(str(num_checks)+" "+str(num_bits)+"\n")
        
        for row_index in range(len(adj_list)):
            row_list = list(adj_list[row_index])
            
            for entry_index in range(len(row_list)):
                entry = row_list[entry_index]
                file.write("%s" % entry)
                
                # Insert a space between entries, except for the last entry.
                if ((entry_index+1) < len(row_list)):
                    file.write(" ")
                    
            # Insert a new line character between lines, except for the last line.
            if ((row_index+1) < len(adj_list)):
                file.write("\n")






# Function to eliminate multi-edges in a Tanner graph by reassigning connections.
# Preserves the column weight and row weight in the parity check matrix H

# Function Inputs:
# H_input: the LDPC parity check matrix corresponding to a Tanner graph

# Functiount Ouputs:
# H_ouput: a possibly modified version of the input matrix with no remaining multi-edges.

def reassign_multi_edges(H_input):
    
    # Infer the dimensions of the input matrix.
    num_rows, num_cols = H_input.shape
    
    # Make a copy of the input matrix to modify while swapping edges; this matrix will be returned at the end of the function.
    H_output = np.copy(H_input)
    
    # Loop over every position in the matrix.
    for row1,col1 in ((i,j) for i in range(num_rows) for j in range(num_cols)):
        # An entry of weight > 1 indicates a multi-edge
        while (H_output[row1,col1] > 1):
            # Loop through other positions in the matrix until a suitable swap can be identified.
            # This search is reapted with a while-loop until the weight of this entry is 1.
            for row2,col2 in ((i,j) for i in range(num_rows) for j in range(num_cols)):
                if ((H_output[row1,col2]==0) and (H_output[row2,col1]==0) and (H_output[row2,col2]>0)):
                    # If this forms a "square" with another entry meeting the above conditions, an edge swap is possible.
                    # These conditions ensure that the column weight and row weight remain constant.
                    H_output[row1,col1] -= 1
                    H_output[row2,col2] -= 1
                    H_output[row1,col2] += 1
                    H_output[row2,col1] += 1
                    
                    # Break out of the current for-loop
                    break
                    
    # Return the modified matrix; it should have no remaining multi-edges.
    return H_output



# Function to generate a random parity check matrix with rows and columns of constant weight.
# This works by constructing two halves of the corresponding Tanner graph.
# We choose edges by using a random permutation to match the the nodes between the two halves.
# The bit nodes have constant degree, and as do the check nodes.
# The total degree of each half of this graph must match.

# Function Inputs
# total_bits: the number of nodes in the left-half Tanner graph = the number of columns in H.
# bit_node_deg: the degree of each node in the left-half Tanner graph = the weight of a col in H.
# check_node_deg: the degree of each node in the right-half Tanner graph = the weight of a row in H.

# Function Outputs:
# H: the parity check matrix corresponding to this Tanner graph.

def generate_random_H_matrix(total_bits,bit_node_deg,check_node_deg):
    
    total_deg = total_bits * bit_node_deg
    
    if ( (total_deg%check_node_deg) != 0 ):
        raise Exception("The check node degree is not a divisior of the total degree.")
    else:
        total_checks = total_deg // check_node_deg
        
    # Generate a random permuation of length = total_deg
    permutation = np.random.permutation(total_deg)
    
    # To construct the parity check matrix H, initialize a 0-matrix of the correct dimensions.
    # Loop through the number of bits and checks to populate the entries according to the permutation.
    H = np.zeros((total_checks,total_bits),dtype=int)
    
    for i in range(total_deg):
        col = i%total_bits
        row = permutation[i]%total_checks
        H[row][col] += 1

    # The matrix obtained in this way may have entries which are greater than 1.
    # These correspond to multi-edges in the Tannger graph, which we would prefer to avoid.
    # Reassign possible multi-edges so that each entry is at most 1, while preserving the constant row and column weight.
    H = reassign_multi_edges(H)
        
    return H



# Modified version of the function to generate a random parity check matrix with constant weight rows and columns.
# The "constant weight function" may result in multi-edges (entries of 2 or greater), which are problematic in our computations.
# Instead, we can choose to ignore multi-edges by not allowing matrix entries greater than 1.
# The resulting matrix might not have constant weight rows and columns, but this work around is quick and easy.

# FUNCTION NO LONGER NEEDED THANKS TO THE FUNCTION TO REASSIGN MULTI-EDGES

# Function Inputs
# total_bits: the number of nodes in the left-half Tanner graph = the number of columns in H.
# bit_node_deg: the degree of each node in the left-half Tanner graph = the weight of a col in H.
# check_node_deg: the degree of each node in the right-half Tanner graph = the weight of a row in H.

# Function Outputs:
# H: the parity check matrix corresponding to this Tanner graph.

def generate_random_H_matrix_non_constant_weights(total_bits,bit_node_deg,check_node_deg):
    
    total_deg = total_bits * bit_node_deg
    
    if ( (total_deg%check_node_deg) != 0 ):
        raise Exception("The check node degree is not a divisior of the total degree.")
    else:
        total_checks = total_deg // check_node_deg
        
    # Generate a random permuation of length = total_deg
    permutation = np.random.permutation(total_deg)
    
    # To construct the parity check matrix H, initialize a 0-matrix of the correct dimensions.
    # Loop through the number of bits and checks to populate the entries according to the permutation.
    H = np.zeros((total_checks,total_bits),dtype=int)
    
    for i in range(total_deg):
        col = i%total_bits
        row = permutation[i]%total_checks
        H[row][col] = 1     # This is the only line of code modified from the preceding version of the function.
        
    return H


# Function to idenitfy and count 4-cycles in a Tanner graph using the parity check matrix.
# Based on a portion of the function that breaks these cycles.
# This function is just to help examine a given matrix while testing.

# Function Inputs:
# H_input: the LDPC parity check matrix corresponding to a Tanner graph

# Functiount Ouputs:
# num_4cycles: the number of 4-cycles in the corresponding Tanner graph.

def count_4cycles(H_input):
    
    # Infer the dimensions of the input matrix.
    num_rows, num_cols = H_input.shape
    
    # Make a copy of the input matrix to modify while swapping edges.
    H_output = np.copy(H_input)
    
    # initialize a variable to keep track of the number of 4-cycles counted.
    num_4cycles = 0
    
    # Loop over every position in the matrix; this is the "top-left corner" of a square.
    for row1,col1 in ((i,j) for i in range(num_rows) for j in range(num_cols)):
        # Loop over the remaing positions in the matrix to the right and below this corner.
        # This is the "bottom-right corner" of the square.
        for row2,col2 in ((s,t) for s in range(row1+1,num_rows) for t in range(col1+1,num_cols)):
            # A 4-cycle is identified if each entry in this square is 1.
            if ((H_output[row1,col1]==1) and 
                (H_output[row1,col2]==1) and 
                (H_output[row2,col1]==1) and 
                (H_output[row2,col2]==1)):
                
                # Increment the number of 4-cycles counted in this matrix.
                num_4cycles += 1
                
    # Return the total number of 4-cycles counted in this way.
    return num_4cycles


# A function to construct a random HGP code from two randomly chosen classical parity check matrices.
# At the moment, this uses the non-constant weight row/column version of the function.
# The inputs represent the various parameters of the classicial parity check matrices.
# We have the constraint that n1 and r1 must be divisors of total1, and n2 and r2 must be divisors of total2.
# Returns the hyper graph product code class object

def construct_random_HGP_code(total_classical_bits_1,n1,r1,total_classical_bits_2,n2,r2):

    # Include some exceptions to check the constraints on the parameters.
    if ((total_classical_bits_1%r1) != 0):
        raise Exception("The parameter total_classical_bits_1 must be divisible by r1.")
    if ((total_classical_bits_2%r2) != 0):
        raise Exception("The parameter total_classical_bits_2 must be divisible by r2.")

    H1 = generate_random_H_matrix(total_classical_bits_1,n1,r1)
    H2 = generate_random_H_matrix(total_classical_bits_2,n2,r2)
    
    return HGP_code(H1,H2)



# Function to compute the "radius" of the confidence interval for a performance dictionary.

def conf_int_radius(perf_dict):
    
    # Infer the parameters needed for this formula.
    # Note that "z_parameter" is computed as z(1-alpha/2) = 1.96 for a 95% confidence interval.
    M = perf_dict["failure_rate"]
    n = perf_dict["total_trials"]
    z_parameter = 1.96
    
    return z_parameter*math.sqrt((M*(1-M))/n)



# Function to plot the results of the decoder at various error rates using the list of performance dictionaries.

def plot_erasure_rate_vs_failure_rate(HGP_code,list_of_performance_dicts):

    # Using the last dictionary from this list, infer some of the trials' parameters.
    last_dict = list_of_performance_dicts[-1]
    max_erasure_rate = last_dict["erasure_rate"]
    num_trials = last_dict["total_trials"]
    
    # Determine the x and y values for the plot from the list of performance dictionaries
    x_values_list = [performance_dict["erasure_rate"] for performance_dict in list_of_performance_dicts]
    y_values_list = [performance_dict["failure_rate"] for performance_dict in list_of_performance_dicts]

    plt.plot(x_values_list,y_values_list)[0]
    plt.axis([0,1.1*max(x_values_list),0,1.1*max(y_values_list)])
    plt.xlabel("Erasure Rate")
    plt.ylabel("Failure Rate for Erasure-supported Error Recovery")
    plt.title("Peeling Decoder Performance with ["+str(HGP_code.num_qubits)+","+str(HGP_code.dim)+"] HGP Code")
    plt.annotate('Number of qubits: '+str(HGP_code.num_qubits), xy=(0.05, 0.9), xycoords='axes fraction')
    plt.annotate('Maximum erasure rate: '+str(max(x_values_list)), xy=(0.05, 0.85), xycoords='axes fraction')
    plt.annotate('Number of different erasure rates: '+str(len(x_values_list)), xy=(0.05, 0.8), xycoords='axes fraction')
    plt.annotate('Number of randomized trials per erasure rate: '+str(num_trials), xy=(0.05, 0.75), xycoords='axes fraction')
    plt.show()




# Function to compute the power set of a given set (that is, the set of all possible subsets).
# Taken from: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



# Construct a submatrix by deleting rows and columns with the specified index.

# Function Inputs:
# input_matrix: the full matrix from which we will construct a submatrix
# row_index_set: a set/tuple/etc of indices for the rows to be deleted from this matrix
# col_index_set: a set/tuple/etc of indices from the columns to be deleted from this matrix

# Function Outputs:
# submatrix: the desired submatrix of the input matrix.

def find_submatrix(input_matrix,row_index_set,col_index_set):
    
    submatrix = np.copy(input_matrix)
    
    # Sort the the row and column indices so that they are lists of decreasing indices.
    # This guarantees that higher-index rows and columns will be deleted first, avoiding re-indexing errors.
    sorted_rows = sorted(row_index_set,reverse=True)
    sorted_cols = sorted(col_index_set,reverse=True)
    
    # Loop to delete the rows with the given indices
    for row_index in sorted_rows:
        submatrix = np.delete(submatrix,row_index,axis=0)
        
        # DEBUG
        #print(submatrix)
        
    # Loop to delete the columns with the given indices
    for col_index in sorted_cols:
        submatrix = np.delete(submatrix,col_index,axis=1)
        
        # DEBUG
        #print(submatrix)
    
    return submatrix



# Function to exhaustively search for a sufficiently large submatrix of 1s.
# Considers larger matrices first, and terminates search when remaining matrices are too small.

# Function Inputs:
# input_matrix: the matrix of 0s and 1s to search through
# critical_size_B: a constant representing the minimum required size of the set B of horizontal qubits.
# critical_size_A: a constant representing the minimum required size of the set A of vertical qubits.

# Function Outputs:
# existence_bool: a boolean value, True if a large enough submatrix exists, and False otherwise.
# B_row_indices: the set of row indices for the last computed set B
# A_col_indices: the set of column indices for the last computed set A

# NOTE: Must confirm that the row and column indices of the matrix in question correspond to the qubit indices;
# I think these are different in the matrix that gets passed to this function, but the original indices can be identified.

def search_for_critical_submatrix(input_matrix,critical_size_B,critical_size_A):
    
    # Infer the dimensions of the input matrix
    num_rows, num_cols = input_matrix.shape
    
    # Initialize a boolean to track whether we find such a matrix, and sets of row and column indices for the sets B and A.
    existence_bool = False
    B_row_indices = set()
    A_col_indices = set()
    
    # Compute lists of possible row and column indices to delete from the input matrix while searching for a suitable submatrix.
    row_index_set_list = list(powerset(range(num_rows)))
    col_index_set_list = list(powerset(range(num_cols)))
    
    # DEBUG:
    #print("Row index set list:",row_index_set_list)
    #print("Col index set list:",col_index_set_list)
    
    # Iterate through pairs of index sets of rows and columns to be deleted from the input matrix.
    for row_index_set,col_index_set in ((i,j) for i in row_index_set_list for j in col_index_set_list):
        # We exclude deleting more rows/columns than the critical size of the sets B and A require.
        if ( ((num_rows-len(row_index_set)) >= critical_size_B) and ((num_cols-len(col_index_set)) >= critical_size_A) ):
            # Compute a possible submatrix by deleting the specified rows and columns
            submatrix = find_submatrix(input_matrix,row_index_set,col_index_set)
            
            # DEBUG
            #print("Submatrix from deleting rows",row_index_set,"and columns",col_index_set)
            #print(submatrix)
            
            # If this submatrix consists entirely of 1s, the criticality condition is satisfied.
            # Set the existence booelan to true and break the for loop for this search.
            if (np.count_nonzero(submatrix) == submatrix.size):
                existence_bool = True
                break
    
    # If a sufficiently large submatrix was found, we must infer the corresponding row and column indices.
    # These may be determined as the complement of the LAST sets of deleted row and column indices.
    if existence_bool:
        B_row_indices = set(range(num_rows)).difference(set(row_index_set))
        A_col_indices = set(range(num_cols)).difference(set(col_index_set))
      
    # Return this submatrix, and the corresponding row/column indices that define it.
    return existence_bool, B_row_indices, A_col_indices



# Small function to compute the set of checks adjacent to a set of given qubits.

# Function Inputs:
# HGP_code: the hypergraph product code class object for the code under consideration.
# qubit_index_set: any set of qubit indices for this code.

# Function Outputs:
# adjacent_check_index_set: the set of indices of checks adjacent to any of the qubits in the given set.

def compute_set_of_checks_adjacent_to_given_qubits(HGP_code,qubit_index_set):
    
    adjacent_check_index_set = set()
    for qubit_index in qubit_index_set:
        adjacent_check_index_set.update(HGP_code.list_of_checks_per_qubit[qubit_index])
        
    return adjacent_check_index_set



# NOTE: Same as the function above; re-written by mistake.
# Since both functions are used in various places, it easier just to keep both.

# Stand-alone function to compute the set of checks adjacent to a given set of qubits.
# Used as part of the connected components/clusters in the erasure pattern.
# This way, we can work with the qubit index set without explicitly tracking the checks.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# E_index_set: the set of erased qubit indices in a given stopping set for this code.

# Function Outputs
# Ch_index_set: the set of check indices adjacent to these erased qubits.

def compute_adjacent_check_indices(HGP_code,E_index_set):
    
    # Initialize an empty set of adjacent check indices; this set will be returned at the end of this function.
    Ch_index_set = set()
    
    # The set of adjacent check indices can be inferred by union-ing "list_of_checks_per_qubit" sets from the HGP code.
    for qubit_index in E_index_set:
        Ch_index_set.update(HGP_code.list_of_checks_per_qubit[qubit_index])
    
    # Return the finished set of adjacent check indices.
    return Ch_index_set


# Function to divide a given subgraph of the Tanner graph of a HGP code into connected components.
# Ideally, this is used to decompose a stopping set into disjoint classical stopping sets?
# For given set of (erased) qubits, we consier the induced subgraph of the Tanner graph using all adjacent checks.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# E_index_set: the set of erased qubit indices in a given stopping set for this code.
# Ch_index_set: the set of check indices adjacent to these erased qubits.

# Function Outputs:
# num_connected_components: the number of connected components in this subgraph.
# list_of_E_comp_index_sets: a list of sets of qubit indices for each such connected component.
# list_of_Ch_comp_index_sets: a list of sets of check indices for each such connected component.

def decompose_subgraph_into_connected_components(HGP_code,E_index_set,Ch_index_set):
    
    # Initialize the sets that will be returned by this function.
    num_connected_components = 0
    list_of_E_comp_index_sets = []
    list_of_Ch_comp_index_sets = []
    
    # Create copies of the given sets of qubits and checks that will be divided into pieces.
    E_local = E_index_set.copy()
    Ch_local = Ch_index_set.copy()
    
    # We will remove qubits from E_local until the set is exhausted.
    while (E_local != set()): 
        E_temp_list = [E_local.pop()]
        i = 0
        # Loop through the qubit indices in a single connected component, while building up that component.
        while(i < len(E_temp_list)):
            q_temp_index = E_temp_list[i]
            for adj_check in HGP_code.list_of_checks_per_qubit[q_temp_index]:
                # Loop through those qubits a distance 2 from the initial qubit.
                for adj_qubit in HGP_code.list_of_qubits_per_check[adj_check]:
                    # If any such qubit is contained in E_local, remove it and add it to E_temp_list
                    if (adj_qubit in E_local):
                        E_local.discard(adj_qubit)
                        E_temp_list.append(adj_qubit)           
            i += 1
                        
        # Convert the list of qubits in this component into a set, and add it to the list of sets.
        list_of_E_comp_index_sets.append(set(E_temp_list))
        num_connected_components += 1
        
    # Next, compute the sets of adjacent checks for each of these connected components.
    for E_comp_index_set in list_of_E_comp_index_sets:
        Ch_temp_set = set()
        for qubit_index in E_comp_index_set:
            for adj_check in HGP_code.list_of_checks_per_qubit[qubit_index]:
                Ch_temp_set.add(adj_check)
        list_of_Ch_comp_index_sets.append(Ch_temp_set)
        
    return num_connected_components, list_of_E_comp_index_sets, list_of_Ch_comp_index_sets


# Function to distinguish classical stopping sets from non-classical stopping sets.
# A stopping set is determined to be classical if it is entirely supported on a single row or column of qubits in the HGP code.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# E_index_set: the set of erased qubit indices in a given stopping set for this code.
# Ch_index_set: the set of check indices adjacent to these erased qubits.

# Function Outpus:
# True if this stopping set is classical, and False otherwise.

def is_classical_stopping_set(HGP_code,E_index_set,Ch_index_set):
    
    # Decompose the set of erased qubits into connected components.
    num_comps,list_of_E_comp_index_sets,list_of_Ch_comp_index_sets = decompose_subgraph_into_connected_components(HGP_code,E_index_set,Ch_index_set)
    
    # Handle each connected component separately.
    list_of_E_biindex_sets = []
    list_of_Ch_biindex_sets = []
    list_of_classical_stopping_set_bools = []
    
    # Loop through the number of classical components.
    for i in range(num_comps):
        # Initilize sets to represent the biindices of the qubits in E.
        E_biindex_set = set()
        Ch_biindex_set = set()

        # Convert the input set of erased qubit indices into the corresponding set of qubit biindices.
        for qubit_index in list_of_E_comp_index_sets[i]:
            # There are two cases to consider for horizontal and vertical qubits.
            # These are distinguished by the qubit index.
            if (qubit_index < HGP_code.num_h_qubits):
                # Horizontal case:
                qubit_biindex = index_to_biindex(qubit_index,HGP_code.n2,0)
            else:
                # Vertical case:
                qubit_biindex = index_to_biindex(qubit_index,HGP_code.r2,HGP_code.num_h_qubits)
            E_biindex_set.add(qubit_biindex)

        for check_index in list_of_Ch_comp_index_sets[i]:
            check_biindex = index_to_biindex(check_index,HGP_code.r2)
            Ch_biindex_set.add(check_biindex)

        # Initialize sets to represent the first and second components of the qubit biindices.
        # If all qubits are in the same row or column, then one of these should be a singleton set.
        E_biindex_1_set = set()
        E_biindex_2_set = set()

        for qubit_biindex in E_biindex_set:
            E_biindex_1_set.add(qubit_biindex[0])
            E_biindex_2_set.add(qubit_biindex[1])

        # DEBUG
        #print("E biindex set:",E_biindex_set)
        #print("E biindex 1 set:",E_biindex_1_set)
        #print("E biindex 2 set",E_biindex_2_set)
        #print("Ch biindex set",Ch_biindex_set)
        
        list_of_E_biindex_sets.append(E_biindex_set)
        list_of_Ch_biindex_sets.append(Ch_biindex_set)

        if ((len(E_biindex_1_set)==1) or (len(E_biindex_2_set)==1)):
            list_of_classical_stopping_set_bools.append(True)
        else:
            list_of_classical_stopping_set_bools.append(False)
            
    # The original stopping set is only a classical stopping set if all of the connected components are.
    # The "all()" Python function will return True if and only if all elements in a list are True, and False otherwise.
    return all(list_of_classical_stopping_set_bools)



# Function to decompose a connected component of an erasure pattern into horizontal and vertical clusters of erased qubits.
# A horizontal cluster consists of all horizontal qubits in the same row of the HGP construction.
# A vertical cluster consists of all vertical qubits in the same column of the HGP construction.
# This function is similar is implementation to the function to find connected components, but it's not the same.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# E_index_set: the set of erased qubit indices in a given stopping set for this code.
# Ch_index_set: the set of check indices adjacent to these erased qubits.

# Function Outputs:
# num_clusters: the number of clusters of qubits in this erasure pattern.
# list_of_E_cluster_index_sets: the list of sets of qubit indices in each cluster in the erasure pattern.
# list_of_Ch_cluster_index_sets: the list of sets of check indices adjacent to the qubits in the corresponding cluster.

def decompose_component_into_clusters(HGP_code,E_index_set,Ch_index_set):
    
    # We prefer to restrict this particular function to connected components.
    # Raise an exception if it is applied to an erasure pattern which is NOT a single connected component.
    if (decompose_subgraph_into_connected_components(HGP_code,E_index_set,Ch_index_set)[0] != 1):
        raise Exception('Warning, only decompose connected components into clusters; erasure pattern seems to be disconnected.')
        
    # Make a copy of the set of erased qubit indices, and then decompose this into clusters.
    E_index_set_local = E_index_set.copy()
    
    # Initialize the sets that will be returned by this function.
    num_clusters = 0
    list_of_E_cluster_index_sets = []
    list_of_Ch_cluster_index_sets = []
    
    while (E_index_set_local != set()):
        # Pop a qubit index to identify the corresponding cluster.
        initial_qubit_index = E_index_set_local.pop()
        local_cluster_E_index_set = {initial_qubit_index}
        local_cluster_Ch_index_set = set()
        
        # There are two cases consider depending on if this qubit is horizontal or vertical.
        if (initial_qubit_index < HGP_code.num_h_qubits):
            # Horizontal case:
            initial_qubit_biindex = index_to_biindex(initial_qubit_index,HGP_code.n2,0)
            for next_qubit_index in E_index_set_local:
                # If this qubit is also horizontal:
                if (next_qubit_index < HGP_code.num_h_qubits):
                    next_qubit_biindex = index_to_biindex(next_qubit_index,HGP_code.n2,0)
                    # If the first component of the biindex matches, then qubits are in the same row.
                    if (next_qubit_biindex[0] == initial_qubit_biindex[0]):
                        local_cluster_E_index_set.add(next_qubit_index)
                    
        else:
            # Vertical case:
            initial_qubit_biindex = index_to_biindex(initial_qubit_index,HGP_code.r2,HGP_code.num_h_qubits)
            for next_qubit_index in E_index_set_local:
                # If this qubit is also vertical:
                if (next_qubit_index >= HGP_code.num_h_qubits):
                    next_qubit_biindex = index_to_biindex(next_qubit_index,HGP_code.r2,HGP_code.num_h_qubits)
                    # If the second component of the biindex matches, then qubits are in the same column.
                    if (next_qubit_biindex[1] == initial_qubit_biindex[1]):
                        local_cluster_E_index_set.add(next_qubit_index)
        
        # Remove the quibts so far identified in the same row or column from the local E index set.
        E_index_set_local.difference_update(local_cluster_E_index_set)
        
        # After extracting the subset of erased qubits in the same row or column as the first, identify the adjacent checks.
        for qubit_index in local_cluster_E_index_set:
            # Union the sets of adjacent check indices for each qubit in this local cluster.
            local_cluster_Ch_index_set.update(HGP_code.list_of_checks_per_qubit[qubit_index])
            
        # The row or column extracted in this way may not be a single cluster.
        # It is possible that a single row or column from the same connected component splits into two disconnected clusters.
        # These can be further identified using the function which extract connected components.
        cluster_components = decompose_subgraph_into_connected_components(HGP_code,
                                                                          local_cluster_E_index_set,
                                                                          local_cluster_Ch_index_set)
        # Use these to infer the number of additional clusters added.
        num_clusters += cluster_components[0]
        for i in range(cluster_components[0]):
            list_of_E_cluster_index_sets.append(cluster_components[1][i])
            list_of_Ch_cluster_index_sets.append(cluster_components[2][i])
    
    return num_clusters, list_of_E_cluster_index_sets, list_of_Ch_cluster_index_sets



# Function to construct a list of sets of shared checks per cluster from a list of clusters in a decomposition of a component.
# The checks are indexed according a given list of clusters.
# Also constructs the corresponding list of sets of clusters per check (these could be empty, singleton, or size 2).

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# list_of_E_cluster_index_sets: the list of sets of qubit indices in each cluster in a connected comp. of the erasure pattern.

# Function Outputs:
# list_of_shared_checks_per_cluster: a list of sets of indices for checks that this cluster shares with other clusters.
# list_of_clusters_per_check: a list of indices of clusters adjacent to a given check.

def compute_list_of_shared_checks_per_cluster_and_vice_versa(HGP_code,list_of_E_cluster_index_sets):
    
    # Infer the number of clusters.
    num_clusters = len(list_of_E_cluster_index_sets)
    
    # Infer the list of sets of adjacent check indices.
    list_of_Ch_cluster_index_sets = []
    for i in range(num_clusters):
        list_of_Ch_cluster_index_sets.append(compute_adjacent_check_indices(HGP_code,list_of_E_cluster_index_sets[i]))
        
    # DEBUG
    #print("E cluster list:",list_of_E_cluster_index_sets)
    #print("Ch cluster list:",list_of_Ch_cluster_index_sets)
    
    # Initialize lists to return at the end of this function.
    list_of_shared_checks_per_cluster = []
    list_of_clusters_per_check = []
    
    # Loop through the clusters and determine which checks, if any, are shared with another cluster.
    for cluster_index in range(num_clusters):
        Ch_cluster_index_set = list_of_Ch_cluster_index_sets[cluster_index]
        cluster_shared_checks = set()
        
        # Loop through the other clusters and see whether they overlap at all with this cluster index set.
        for other_cluster_index in [i for i in range(num_clusters) if i != cluster_index]:
            Ch_other_cluster_index_set = list_of_Ch_cluster_index_sets[other_cluster_index]
            # If the two cluster index sets have a non-empty intersection, then they are a check in common.
            shared_checks_temp = Ch_cluster_index_set.intersection(Ch_other_cluster_index_set)
            # Add any check indices in this intersection to the set of shared checks for this cluster.
            # Nothing happens if the intersection is empty.
            for shared_check_index in shared_checks_temp:
                cluster_shared_checks.add(shared_check_index)
                
        # Append this set of shared checks to the list of shared checks per cluster.
        list_of_shared_checks_per_cluster.append(cluster_shared_checks)
        
    # To determine the list of checks per cluster, loop through the number of clusters from the HGP code.
    for check_index in range(HGP_code.num_checks):
        clusters_adjacent_to_this_check = set()
        # Loop through the set of clusters' checks, and verify whether it contains this check.
        for cluster_index in range(num_clusters):
            cluster_Ch_index_set = list_of_Ch_cluster_index_sets[cluster_index]
            # If this cluster does contain the check, then add the cluster index to the set of adjacent clusters.
            if (check_index in cluster_Ch_index_set):
                clusters_adjacent_to_this_check.add(cluster_index)
        
        # Append this list of clusters per check to the list.
        list_of_clusters_per_check.append(clusters_adjacent_to_this_check)
        
    # Return both lits computed in this way.
    return list_of_shared_checks_per_cluster, list_of_clusters_per_check



# Function meant to perform syndrome analysis for a classical code using Gaussian elimination.
# That is, given a parity check matrix and a sydrome vector, find a corresponding error vector with this syndrome.
# If more than one solution is possible, choose one with the smallest weight.

# This function also accounts for an erasure pattern.
# By default, every non-erased qubit must be left as 0 in the predicted error vector.
# To achieve this, the columns of the parity check matrix corresponding to this vector are first "zeroed" before proceeding.
# Hence, any solution which is found does not use the columns; the rest of the function can proceed as before.

# Function Inputs:
# H: the parity check matrix of the classical code.
# s: the syndrome vector of the classical code.
# E_index_set: the set of indices for the erased bits; columns of H for bits not in this set are "zeroed".

# Function Outputs:
# predicted_e: a predicted error vector e of minimal weight satisfying H*e = s.

def perform_classical_syndrome_analysis_with_erasure(H,s,E_index_set):
    
    # DEBUG
    #print("Original parity check matrix, syndrome vector, and erasure index set:")
    #print("H = ")
    #print(H)
    #print("s = ",s)
    #print("E_index_set = ",E_index_set)
    
    # Verify that the length of the syndrome vector matches the number of rows in the parity check matrix.
    if (np.shape(H)[0] != len(s)):
        raise Exception('The length of the syndrome vector does not match the number of rows of the parity check matrix.')
        
    # Infer the number of bits from the number of columns of the parity check matrix.
    num_bits = np.shape(H)[1]
    
    # Construct a modified parity check matrix by setting to 0 any column of H corresponding to a non-erased bit.
    H_zeroed = H.copy()
    for bit_index in range(num_bits):
        if (bit_index not in E_index_set):
            H_zeroed[:,bit_index] = 0
            
    # DEBUG
    #print("H_zeroed matrix after zeroing columns of H corresponding to non-erased bits:")
    #print(H_zeroed)
        
    # Construct the augmented matrix [H|s] corresponding to this system of equations.
    H_aug = np.hstack((H_zeroed,s[:,np.newaxis]))
    
    # DEBUG:
    #print("Augmented matrix:")
    #print(H_aug)
    
    # Place this augmented matrix into Reduced Row Echelon Form over GF(2).
    H_aug_rref, A, pivot_indices = standard_form(H_aug)
    
    # DEBUG
    #print("Augmented RREF:")
    #print(H_aug_rref)
    #print("Pivot column indices:",pivot_indices)
    
    # Split this augmented RREF matrix into the component from H and the vector from s.
    H_rref = H_aug_rref[:,:-1]
    s_rref = H_aug_rref[:,-1]
    
    # DEBUG:
    #print("H_rref:")
    #print(H_rref)
    #print("s_rref",s_rref)
    
    # Every non-pivot entry is a free variable.
    # To choose a solution of minimal weight, we will just assume that all pivot variables are 0. (is it really that simple?)
    # Do this by initializing a 0-vector of the appropriate length, and looping backwards through the variables.
    # Then just assign the pivot indices to match the corresponding syndrome values.
    predicted_e = np.zeros(np.shape(H)[1],dtype=int)
    for i in reversed(range(len(pivot_indices))):
        pivot_index = pivot_indices[i]
        predicted_e[pivot_index] = s_rref[i]
        
    # DEBUG
    #print("Predicted error vector:",predicted_e)
    
    # DEBUG
    #print("Check that the predicted error vector matches the input syndrome: H*v =? s")
    #print("H*v:",np.dot(H,predicted_e)%2)
    #print("s:",s)
    
    # This should always yield a solution, but if I messed up and it doesn't, flag this.
    if (not np.array_equal(np.dot(H,predicted_e)%2,s)):
        raise Exception("Something went wrong! The predicted error vector does not yield the given syndrome.")
        
    return predicted_e



# Function to load a classical parity check matrix from a text file and construct a corresponding HGP class object.
# By default, this function assumes that H1 = H2 in this particular construction.
# The input text file is hence the adjaceny list for this single classical parity check matrix.
# The first line of the text file is not part of the matrix; rather, it indicates the number of classical checks and bits.
# We assume that the textfile to be imported is in the same working directory as this function.

# Function Inputs:
# H_adj_list_file_name: a string representing the name of the text file to import.

# Function Outputs:
# HGP_code: the Hypergraph Product Code class object constructed using this classical parity check matrix.

def construct_HGP_code_from_classical_H_text_file(H_adj_list_file_name):
    
    # Before proceeding, verify that the input file name is a string.
    if (type(H_adj_list_file_name) != str):
        raise Exception('Must provide a string of the filename for the classical H matrix to construct a HGP code.')
    
    # Infer the number of checks and bits of the input classical parity check matrix.
    num_checks, num_bits = np.loadtxt(H_adj_list_file_name,dtype=int,max_rows=1)
    
    # Infer the adjacency list of this matrix from the rest of the file.
    # NOTE: This function only works for matrices describing a Tanner graph with constant bit degree and constant check degree.
    #adj_list = np.loadtxt(H_adj_list_file_name,dtype=int,skiprows=1)

    # Modified version of the function to read in an adjacency list, allowing for non-constant degree nodes.
    adj_list = []
    for i in range(num_checks):
        adj_list_row = np.loadtxt(H_adj_list_file_name,dtype=int,skiprows=i+1,max_rows=1)
        adj_list.append(set(adj_list_row))

    
    # Convert this adjacency list into a binary matrix (this is the parity check matrix of the classical code).
    H = convert_adjacency_list_to_binary_matrix(num_checks,num_bits,adj_list)
    
    # Initialize a Hypergraph Product Code class object using this classical matrix as H1 = H2.
    HGP_code_class_object = HGP_code(H,H)
    
    # Return the HGP code class object.
    return HGP_code_class_object