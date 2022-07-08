import numpy as np


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



# Place a matrix in standard form.

def standard_form(M_input):
    #### M is an m*n matrix of rank r.
    #### After applying gf2elim, up to renaming column indices:
    #### M is made of an identity block of size r and arbitrary entries in a block of size r*(n - r).
    #### Instead of renaming indices, we return a mask corresponding to the identity block: column_indices_of_pivots.
    #### The block with arbitrary entries corresponds to the other column indices.
    
    # Create a copy to avoid modifying the input matrix.
    M = np.copy(M_input)
    m, n = M.shape

    i = 0
    j = 0

    column_indices_of_pivots = []

    while i < m and j < n:
        #if i%100==0: print('nb rows in echelon form:', i)
        #### find index of largest entry (i.e. 1) in remainder of column j.
        k = np.argmax(M[i:, j]) + i
        #### If the max is O, col is zero and this iteration does nothing.
        #### At the end of the algorithm a full row will be zero (except maybe free variables).
        #### This should be taken into account.
        pivot_value = M[k, j]

        if pivot_value == 0:
            j += 1
        
        else:
            column_indices_of_pivots.append(j)

            #### swap rows
            temp = np.copy(M[k])
            M[k] = M[i]
            M[i] = temp

            aijn = M[i, j:]                 #### aijn is the end of row i : from j to n.
            col = np.copy(M[:, j])          #### make a copy otherwise M will be directly affected
            col[i] = 0                      #### avoid xoring pivot row with itself
            flip = np.outer(col, aijn)      #### flip is an m by (n-j) block on the right of M. Its i_th row is zero.
            M[:, j:] = M[:, j:] ^ flip      #### ^ is an entrywise xor.
            
            i += 1
            j += 1
    
    #### get rid of zero rows at the bottom of M
    for i in range(m-1, -1, -1):            #### from m-1 down to 0 both included.
        if np.any(M[i,:]):
            M_standard = M[:i+1,:]
            break
    
    #### isolate the non identity block A
    A_transpose = []
    for j in range(n):
         if j not in column_indices_of_pivots:
             A_transpose.append(M_standard[:,j])
    A = np.transpose(np.array(A_transpose))
    
    return M_standard, A, column_indices_of_pivots


# Define a canonical basis vector

def canonical_basis_vector(i, n):
    e = np.zeros(n, dtype=int)
    e[i] = 1
    return e


# Compute a generator matrix based on the standard form of the parity check matrix.
# Used along with the output from the standard_form function.

def compute_G(A, mask, nb_qubits):
    #### mask contains the column indices of the pivots
    r_x = len(mask)
    G_as_list = []
    i_A = 0
    for i in range(nb_qubits):
        if i in mask:
            #### pivot columns of H correspond to non-pivot rows of G
            G_as_list.append(A[i_A,:])
            i_A += 1
            
            # DEBUG
            #print(i,"in mask:",G_as_list)
            
        else:
            #### non-pivot columns of H correspond to pivot rows of G
            G_as_list.append(canonical_basis_vector(i-i_A, nb_qubits-r_x))
            
            # DEBUG
            #print(i,"not in mask:",G_as_list)
            
    return np.array(G_as_list)




# Define a class object to represent a check.
# A list of check objects are stored within the hypergraph product code.

class Check:
    
    def __init__(self, index, biindex, horizontal_qubits, vertical_qubits):
        
        self.index = index
        self.biindex = biindex
        
        # Represented as sets of column indices from the parity check matrix
        self.horizontal_qubits = horizontal_qubits
        self.vertical_qubits = vertical_qubits
        
        # Initialize empy sets of erased qubits; these sets are for use with the expander code.
        self.horizontal_erased_qubits = set()
        self.vertical_erased_qubits = set()
        
    def initialize_erased_qubits(self,erasure_index_set):
        # The erased qubits are determined using the intersections
        self.horizontal_erased_qubits = self.horizontal_qubits.intersection(erasure_index_set)
        self.vertical_erased_qubits = self.vertical_qubits.intersection(erasure_index_set)
        return
        
    def is_dangling(self):
        return (len(self.horizontal_erased_qubits.union(self.vertical_erased_qubits)) == 1)
    
    def is_critical(self):
        return ((len(self.horizontal_erased_qubits)==1) and (len(self.vertical_erased_qubits)==1))
    
    def remove_erased_qubit(self,qubit_index_to_remove):
        # Remove the index of a single qubit from the set of erased qubits; this could be either horizontal or vertical.
        # If the index to remove for some reason is not in this set, then nothing changes.
        self.horizontal_erased_qubits.discard(qubit_index_to_remove)
        self.vertical_erased_qubits.discard(qubit_index_to_remove)
        
    def add_erased_qubits(self,set_of_qubit_indices_to_add):
        # Union of qubits to add to the erasure, but ONLY if the added erased qubits are adjacent to this check.
        # Again, these could be either horizontal or vertical, so we check both.
        self.horizontal_erased_qubits = self.horizontal_erased_qubits.union(
            self.horizontal_qubits.intersection(set_of_qubit_indices_to_add))
        self.vertical_erased_qubits = self.vertical_erased_qubits.union(
            self.vertical_qubits.intersection(set_of_qubit_indices_to_add))
        
        
# Define a similar class object to represent a generator.
# A list of check objects are stored within the hypergraph product code.
# MODIFY TO DO STUFF WITH BREAKING A GENERATOR

class Generator:
    
    def __init__(self, index, biindex, horizontal_qubits, vertical_qubits):
        
        self.index = index
        self.biindex = biindex
        
        # Represented as sets of column indices from the parity check matrix
        self.horizontal_qubits = horizontal_qubits
        self.vertical_qubits = vertical_qubits
        
        # Initialize empy sets of erased qubits; these sets are for use with the expander code.
        self.horizontal_erased_qubits = set()
        self.vertical_erased_qubits = set()
        
    def initialize_erased_qubits(self,erasure_index_set):
        # The erased qubits are determined using the intersections
        self.horizontal_erased_qubits = self.horizontal_qubits.intersection(erasure_index_set)
        self.vertical_erased_qubits = self.vertical_qubits.intersection(erasure_index_set)
        return
    
    def remove_erased_qubit(self,qubit_index_to_remove):
        # Remove the index of a single qubit from the set of erased qubits; this could be either horizontal or vertical.
        # If the index to remove for some reason is not in this set, then nothing changes.
        self.horizontal_erased_qubits.discard(qubit_index_to_remove)
        self.vertical_erased_qubits.discard(qubit_index_to_remove)
        
    def add_erased_qubits(self,set_of_qubit_indices_to_add):
        # Union of qubits to add to the erasure, but ONLY if the added erased qubits are adjacent to this check.
        # Again, these could be either horizontal or vertical, so we check both.
        self.horizontal_erased_qubits = self.horizontal_erased_qubits.union(
            self.horizontal_qubits.intersection(set_of_qubit_indices_to_add))
        self.vertical_erased_qubits = self.vertical_erased_qubits.union(
            self.vertical_qubits.intersection(set_of_qubit_indices_to_add))
    
    def is_entirely_erased_generator(self):
        # Function to determine whether all qubits adjacent to this generator are erased.
        if ( (self.horizontal_qubits == self.horizontal_erased_qubits) and (self.vertical_qubits == self.vertical_erased_qubits) ):
            return True
        else:
            return False



# A class used to define the HGP code construction using the parity check matrices of two classical codes.

class HGP_code:
    
    def __init__(self, H1, H2):
        
        # The parity check matrcies of the two classicial codes used in this construction.
        self.H1 = H1
        self.H2 = H2
        
        # The number of bits and checks from the two classicial codes.
        self.n1 = np.shape(H1)[1]   # The number of bits in C1 = number of columns in H1
        self.r1 = np.shape(H1)[0]   # The number of checks in C1 = number of rows in H1
        self.n2 = np.shape(H2)[1]   # The number of bits in C2 = number of columns in H2
        self.r2 = np.shape(H2)[0]   # The number of checks in C2 = number of rows in H2
        
        self.num_qubits = self.n1*self.n2 + self.r1*self.r2
        
        # The components of the quantum parity check matrices Hx and Hz.
        self.Hx1 = np.kron(H1, np.eye(self.n2,dtype=int))
        self.Hx2 = np.kron(np.eye(self.r1,dtype=int), np.transpose(H2))
        self.Hx = np.hstack((self.Hx1,self.Hx2))
        
        self.Hz1 = np.kron(np.eye(self.n1,dtype=int), H2)
        self.Hz2 = np.kron(np.transpose(H1), np.eye(self.r2,dtype=int))
        self.Hz = np.hstack((self.Hz1,self.Hz2))
        
        # Compute the generator matrices corresponding to Hx and Hz.
        self.Hx_standard_form = standard_form(self.Hx)
        self.Gx = compute_G(self.Hx_standard_form[1],self.Hx_standard_form[2],self.num_qubits).T
        self.Hz_standard_form = standard_form(self.Hz)
        self.Gz = compute_G(self.Hz_standard_form[1],self.Hz_standard_form[2],self.num_qubits).T
        
        # Determine the dimensions of the codes Cx and Cz based on the number of pivots in the parity check matrices.
        # The number of pivots from Hx gives the dimension of the dual code, from which we can determine the dimension of Cx.
        self.kx = self.num_qubits - len(self.Hx_standard_form[2])
        self.kz = self.num_qubits - len(self.Hz_standard_form[2])
        self.dim = self.kx + self.kz - self.num_qubits
        
        
        # Determine adjacency lists corresponding to the matrices Gx and Gz.
        # These are used for working with sparse representations effeciently (for example, to identify logical errors).
        self.Gx_adj_list = compute_adjacency_list(self.Gx)
        self.Gz_adj_list = compute_adjacency_list(self.Gz)
        
        
        # NEW ATTRIBUTES ADDED FOR EXPANDER DECODER
        # NOTE: checks/generators are defined relatvie to X-Pauli errors.
        
        # Numbers of qubits, checks, and generators
        self.num_h_qubits = self.n1 * self.n2
        self.num_v_qubits = self.r1 * self.r2
        self.num_checks = self.n1 * self.r2
        self.num_generators = self.r1 * self.n2
        
        # Compute the adjaceny lists indicating connections between qubits and checks and generators.
        # Use the rows of the Hz parity check matrix to identify the sets of indices of adjacent horizontal and vertical qubits.
        self.list_of_horizontal_qubits_per_check = compute_adjacency_list(self.Hz1)
        self.list_of_vertical_qubits_per_check = compute_adjacency_list(self.Hz2,self.num_h_qubits)   # NOTE: shift qubit indices
        self.list_of_qubits_per_check = compute_adjacency_list(self.Hz)
        # Similarly, use the rows of the Hx parity check matrix to identify the sets of idices of adjacent qubits.
        self.list_of_horizontal_qubits_per_generator = compute_adjacency_list(self.Hx1)
        self.list_of_vertical_qubits_per_generator = compute_adjacency_list(self.Hx2,self.num_h_qubits)   # NOTE: shift qubit indices
        self.list_of_qubits_per_generator = compute_adjacency_list(self.Hx)
        # By transposing Hz and Hx, we may also find the sets of indices of checks/generators adjacent to each qubit.
        # These are treated independently from the erasure.
        self.list_of_checks_per_qubit = compute_adjacency_list(self.Hz.T)
        self.list_of_generators_per_qubit = compute_adjacency_list(self.Hx.T)
        
        
        # Initialize lists of check and generator class objects.
        # For each row of the parity check matrix Hz, initialize a corresponding check class object and store these in a list.
        # Initialize a list of indices of those checks which are determined to be dangling.
        self.list_of_checks = []
        self.list_of_generators = []
        
        for i in range(self.num_checks):
            check = Check(i,
                          index_to_biindex(i,self.r2),
                          self.list_of_horizontal_qubits_per_check[i],
                          self.list_of_vertical_qubits_per_check[i])
            self.list_of_checks.append(check)
            
        for i in range(self.num_generators):
            generator = Generator(i,
                                  index_to_biindex(i,self.n2),
                                  self.list_of_horizontal_qubits_per_generator[i],
                                  self.list_of_vertical_qubits_per_generator[i])
            self.list_of_generators.append(generator)


        # Some generators may "overlap" on some of their qubits.
        # The tensor-product of these generators is also a generator, with adjacent qubits determined using the symmetric difference.
        # These products of generators may be needed for certain functions.
        # Initialize a list of "sets of overlapping generator indices" for each generator.
        self.list_of_sets_of_overlapping_generator_indices_per_generator = []

        # Loop over all genarator indices
        for i in range(self.num_generators):
            # Fix a generator
            # Initialize a set of generator indices which overlap with the fixed generator of index i
            overlapping_generator_indices = set()
            # Loop over all qubits adjacent to this generator.
            for adj_qubit_index in self.list_of_qubits_per_generator[i]:
                # Loop over all other generators adjacent to this qubit
                for j in self.list_of_generators_per_qubit[adj_qubit_index]:
                    # This generator overlaps with the fixed generator of index i; add its index to the set.
                    overlapping_generator_indices.add(j)
            # The set of overlapping generator indices will include the fixed generator; remove this index.
            overlapping_generator_indices.discard(i)
            # Finally, append this set of indices to the list of such sets.
            self.list_of_sets_of_overlapping_generator_indices_per_generator.append(overlapping_generator_indices)

        # Next, products of overlapping generators can be indexed by these generators' pairs of indices.
        # Initialize a set of frozensets of pairs of overlapping generator indices to track these.
        self.set_of_pairs_of_overlapping_generator_indices = set()

        # Loop over the lists of sets of overlapping generator indices.
        for i in range(self.num_generators):
            # Loop over the set of generator indices which overlap with the generator of index i.
            for j in self.list_of_sets_of_overlapping_generator_indices_per_generator[i]:
                # Add the immutable frozenset containing these two indices to the set of pairs of overlapping generator indices.
                # Because these are sets, the order of the pair does not matter, and neither do repeats of this pair.
                self.set_of_pairs_of_overlapping_generator_indices.add(frozenset([i,j]))



        
        
    # Functions to compute the syndrome, depending on the type of Pauli error.
    # Assumes the error vector is a standard column vector.

    def Hz_syn_for_X_err(self,err_vect):
        if (len(err_vect) != self.Hz.shape[1]):
            raise Exception("The length of the given error vector does not match the number of columns of Hz.")
        return self.Hz.dot(err_vect.T)%2
    
    def Hx_syn_for_Z_err(self,err_vect):
        if (len(err_vect) != self.Hx.shape[1]):
            raise Exception("The length of the given error vector does not match the number of columns of Hx.")
        return self.Hx.dot(err_vect.T)%2
    
    
    # Function to compute the set of non-zero indices for an X-Pauli error syndrome vector.
    # This is intended to work efficiently with sparse adjacency list representations, rather than compute matrix products.
    # Adds the index of a syndrome bit to the set of indices if the intersection of an error set and a row of the adjacency list is odd.
    # Meant for use with the expander decoder.
    
    def Hz_syn_index_set_for_X_err(self,error_index_set):
        syndrome_bit_index_set = set()
        for i in range(self.num_checks):
            row_index_set = self.list_of_qubits_per_check[i]
            if (len(row_index_set.intersection(error_index_set))%2 != 0):
                syndrome_bit_index_set.add(i)
        return syndrome_bit_index_set
    
    
    # Functions to determine whether a given error is a nontrivial logical error based on whether it commutes with the rows of Gx or Gz.
    # Nontrivial X-logical error:    Hz * v^T = 0   AND   Gx * v^T =/= 0
    # Nontrivial Z-logical error:    Hx * v^T = 0   AND   Gz * v^T =/= 0
    
    def is_non_trivial_X_logical_error(self,err_vect):
        if (len(err_vect) != self.Gx.shape[1]):
            raise Exception("The length of the given error vector does not match the number of columns of Gx.")
        if ((np.array_equal(self.Hz_syn_for_X_err(err_vect),np.zeros(self.Hz.shape[0],dtype=int))) and 
            (np.array_equal((self.Gx.dot(err_vect.T)%2),np.zeros(self.Gx.shape[0],dtype=int)) == False )):
            return True
        else:
            return False
        
    def is_non_trivial_Z_logical_error(self,err_vect):
        if (len(err_vect) != self.Gz.shape[1]):
            raise Exception("The length of the given error vector does not match the number of columns of Gz.")
        if ((np.array_equal(self.Hx_syn_for_Z_err(err_vect),np.zeros(self.Hx.shape[0],dtype=int))) and 
            (np.array_equal((self.Gz.dot(err_vect.T)%2),np.zeros(self.Gz.shape[0],dtype=int)) == False )):
            return True
        else:
            return False
        
        
    # Function to identify a non-trivial X logical error, but instead using the sparse adjacency list representations.
    # In this case, the set of non-zero indices of an error vector is used, rather than the binary error vector itself.
    
    def is_non_trivial_X_logical_error_index_set(self,error_index_set):
        # Only consider if the adjacency list from Gx if the Hz syndrome is zero.
        if (self.Hz_syn_index_set_for_X_err(error_index_set) == set()):
            for i in range(len(self.Gx_adj_list)):
                row_index_set = self.Gx_adj_list[i]
                if (len(row_index_set.intersection(error_index_set))%2 != 0):
                    # In this case, Gx * v^T is nonzero, and hence we must have a logical error.
                    # This condition needs to be true at least once, so we can stop searching after finding one case where this happens.
                    return True
        # Otherwise, this error vector is not a non-trivial X logical error.
        return False


    # Function to determine whether the product of two overlapping generators is fully erased or not.
    # Although written with the intention of applying to a pair of overlapping generators, this function should generalize as written.
    # That is, given any set of generator indices, this function applies to the tensor-product of all of them.

    def is_product_of_generators_fully_erased(self,index_set):
        # The index_set should be a set object containing two (or more) indices, but a set being unordered prevents easy access.
        # Loop through this set to create a list of these generator objects by accessing their indices from the index set.
        # Because axiom of choice.
        gen_prod_list = []
        for index in index_set:
            gen_prod_list.append(self.list_of_generators[index])
        
        # Initialize lists to represent the sets of qubit indices and erased qubit indices for each of these generators.
        gen_qubit_indices_set_list = []
        gen_erased_qubit_indices_set_list = []
        
        # Infer the indices of the qubits and erased qubits in each of these generators.
        # I will use copies for everything because I'm paranoid about chaning one of the original sets by mistake.
        for generator in gen_prod_list:
            gen_qubit_indices_set_list.append(generator.horizontal_qubits.union(generator.vertical_qubits).copy())
            gen_erased_qubit_indices_set_list.append(generator.horizontal_erased_qubits.union(generator.vertical_erased_qubits).copy())

        # Initialize empty sets to represent the qubit indices and erased qubit indices in the product generator.
        product_generator_qubit_indices = set()
        product_generator_erased_qubit_indices = set()

        # Loop through the list of sets of qubit indices and erased qubit indices.
        # Add these to the product generator sets of indices using the symmetric difference.
        # This corresponds to the tensor-product of all of the generators in this list.
        for qubit_index_set in gen_qubit_indices_set_list:
            product_generator_qubit_indices.symmetric_difference_update(qubit_index_set)
        for erased_qubit_index_set in gen_erased_qubit_indices_set_list:
            product_generator_erased_qubit_indices.symmetric_difference_update(erased_qubit_index_set)

        # The product generator is fully erased if and only if the two sets above are equal.
        if (product_generator_qubit_indices == product_generator_erased_qubit_indices):
            return True
        else:
            return False
        
        
        
# Some sample input for testing

# The parity matrix for the 3-bit repetition code
Hrep3 = np.array([[1,1,0],[0,1,1],[1,0,1]],dtype=int)

# The 3x3 toric code, constructed as a hypergraph product of the repetition code.
Toric3 = HGP_code(Hrep3,Hrep3)

# Example of an X-logical error for the 3x3 toric code
#X_logical_err_vect = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=int)

# Verify that it has 0 syndrome with respect to Hz, but is properly identified as a logical error.
#Toric3.Hz_syn_for_X_err(X_logical_err_vect)
#Toric3.is_non_trivial_X_logical_error(X_logical_err_vect)

