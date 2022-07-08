import numpy as np
import copy
import random
import math
import ast
import sys
import os
from datetime import date

from utilities import index_to_biindex, biindex_to_index
from utilities import generate_random_erasure_and_error_index_sets
from utilities import compute_adjacency_list, convert_adjacency_list_to_binary_matrix
from utilities import compute_set_of_checks_adjacent_to_given_qubits, compute_adjacent_check_indices
from utilities import is_classical_stopping_set, perform_classical_syndrome_analysis_with_erasure
from utilities import decompose_subgraph_into_connected_components, decompose_component_into_clusters
from utilities import compute_list_of_shared_checks_per_cluster_and_vice_versa
from utilities import construct_HGP_code_from_classical_H_text_file

from Hypergraph_Product_Code_Construction_v3 import HGP_code
from Hypergraph_Product_Code_Construction_v3 import standard_form

# Import the 3x3 toric code example for simple testing
from Hypergraph_Product_Code_Construction_v3 import Toric3




# Function to identify the existence of dangling clusters within a connected component.

# Takes a cluster-decomposition of a component as input; assumes that these clusters are actually connected, but does not check.
# Note that we only need to reference the erased qubit indices; the adjacent check indices may be inferred.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# list_of_E_cluster_index_sets: the list of sets of erased qubit indices in a single cluster of a connected component.

# Function Outputs:
# set_of_dangling_cluster_indices: a set of list indices to indicate which clusters are dangling in this connected component.

def identify_dangling_clusters_index_set(HGP_code,list_of_E_cluster_index_sets):
    
    # In the special case where there is only one cluster in the list, this cluster is isolated.
    # This is a special case of a dangling cluster.
    if (len(list_of_E_cluster_index_sets) == 1):
        return {0}
    
    # Initialize a set of dangling cluster indices; this will be returned at the end of the function.
    set_of_dangling_cluster_indices = set()
    
    # Infer the list of adjacent check indices.
    list_of_Ch_cluster_index_sets = []
    for E_cluster_index_set in list_of_E_cluster_index_sets:
        list_of_Ch_cluster_index_sets.append(compute_adjacent_check_indices(HGP_code,E_cluster_index_set))
    
    # A cluster is assumed to be dangling if it shares no more than one check with any other cluster.
    # By this definition, an isolated cluster is also identified as dangling.
    for cluster_index in range(len(list_of_Ch_cluster_index_sets)):
        Ch_cluster_index_set = list_of_Ch_cluster_index_sets[cluster_index]
        # Skip empty clusters, which may be used with other functions.
        if (Ch_cluster_index_set != set()):
            # Initialize a value to keep track of the number of connected clusters.
            num_connected_clusters = 0
            # Loop over the other indices of clusters, skipping this one.
            for other_cluster_index in [i for i in range(len(list_of_Ch_cluster_index_sets)) if i != cluster_index]:
                Ch_other_cluster_index_set = list_of_Ch_cluster_index_sets[other_cluster_index]
                # If the intersection of these two lists of checks is non-empty, then the clusters are connected.
                if (Ch_cluster_index_set.intersection(Ch_other_cluster_index_set) != set()):
                    num_connected_clusters += 1

            # Determine whether this set is dangling based on the total number of connected clusters.
            # If it's not, then add the index of this cluster to the list of dangling clusters.
            if (num_connected_clusters <= 1):
                set_of_dangling_cluster_indices.add(cluster_index)
            
    # Return the set of dangling cluster indices.
    return set_of_dangling_cluster_indices


# Function to determine if a given decomposition of a connected component into clusters can be "peeled".
# That is, can a sequence of dangling clusters be removed to exhaust the clusters in the tree.
# If yes, then we can solve this tree using classical techniques for each dangling cluster.
# If not, then the component contains a cycle of clusters that prevent this.

# Takes a cluster-decomposition of a component as input; assumes that these clusters are actually connected, but does not check.
# Note that we only need to reference the erased qubit indices; the adjacent check indices may be inferred.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# list_of_E_cluster_index_sets: the list of sets of erased qubit indices in a single cluster of a connected component.

# Function Outputs:
# is_peelable: a boolean value stating whether or not this decomposition is peelable; True if a tree, False if contains a cycle.
# peelable_cluster_index_list: a list of indices for clusters which can be peeled; ordered based on order of peeling.

def is_cluster_decomposition_peelable(HGP_code,list_of_E_cluster_index_sets):
    
    # In the special case where there is only one cluster in the list, this cluster is isolated.
    # This is a special case of a dangling cluster, and we will say that this can be peeled.
    if (len(list_of_E_cluster_index_sets) == 1):
        return True, [0]
    
    # Initialize an empty list of clusters which can be peeled; this list will be returned at the end of the function.
    peelable_cluster_index_list = []
    
    # While peeling, we will keep track of a modified list of cluster index sets as clusters are removed.
    # In particular, peeled clusters will be replaced with an empty set in the list (this preserves index order).
    peeled_list_of_E_cluster_index_sets = list_of_E_cluster_index_sets.copy()
    
    # DEBUG
    #print("Initial list of cluster index sets:",peeled_list_of_E_cluster_index_sets)
    
    # Infer the list of adjacent check indices.
    list_of_Ch_cluster_index_sets = []
    for E_cluster_index_set in list_of_E_cluster_index_sets:
        list_of_Ch_cluster_index_sets.append(compute_adjacent_check_indices(HGP_code,E_cluster_index_set))
        
    # Proceed under the assumption that this component is a peelable tree.
    # Run a while loop that terminates when the length of list of peelable indices matches the number of clusters.
    # If the component is determined not to be a tree (if we find no more peelable clusters), then break the loop.
    while (len(peelable_cluster_index_list) < len(list_of_E_cluster_index_sets)):
        # Infer those clusters in the current peeled list of clusters which are dangling.
        local_set_of_dangling_cluster_indices=identify_dangling_clusters_index_set(HGP_code,peeled_list_of_E_cluster_index_sets)
        
        #DEBUG
        #print("Local set of dangling cluster indices:",local_set_of_dangling_cluster_indices)
        
        # If no dangling clusters are identified, break the while loop.
        if (local_set_of_dangling_cluster_indices == set()):
            break
        
        # Add the indices of these peelable clusters to the list.
        # While the global of peelable cluster indices matters, the order of any identified at the same time does not.
        for peelable_cluster_index in local_set_of_dangling_cluster_indices:
            peelable_cluster_index_list.append(peelable_cluster_index)
            
            # Modify the current peeled list of clusters to replace any dangling clusters with empty sets.
            peeled_list_of_E_cluster_index_sets[peelable_cluster_index] = set()
            
            # DEBUG
            #print("Peeled list of E cluster index sets:",peeled_list_of_E_cluster_index_sets)
            
    # The original list of clusters will be peelable as long as the while loop did not terminate early.
    if (len(peelable_cluster_index_list) < len(list_of_E_cluster_index_sets)):
        # In this case, still return the partial list of cluster indices which can be peeled.
        return False, peelable_cluster_index_list
    else:
        return True, peelable_cluster_index_list




# Function attempting to solve a cluster using classical syndrome analysis.
# If a solution does not exist for a given choice of syndrome, report this; this will be used to test possible solutions.

# This function is modified to remove a specified set of checks adjacent to a cluster before performing syndrome analysis.
# This removal corresponds to "zeroing" the corresponding row of the classical parity check matrix before solving.
# The solution found in this way does not depend at all on this removed check (and hence should be easier to solve)
# Notably, zeroing a row preserves the indexing, which is important for the solution we find.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# E_cluster_index_set: the current set of erased qubit indices in JUST this cluster.
# s_full_index_set: the current set of non-zero check indices, corresponding to the "triggered" bits of the FULL syndrome.
# removed_check_indices: a set of indices of checks to be zeroed before solving.

# Function Outputs:
# solution_exists: a boolean value to indicate whether or not a solution exists for this syndrome.
# predicted_cluster_error_vector_index_set: if a solution does exist, the indices of the predicted error vector with this syn.

def cluster_syndrome_analysis_with_zeroing(HGP_code,E_cluster_index_set,s_full_index_set,removed_check_indices):
    
    # Infer the set of adjacent check indices.
    Ch_cluster_index_set = compute_adjacent_check_indices(HGP_code,E_cluster_index_set)
    
    # Infer the portion of the syndrome supported on this set.
    s_cluster_index_set = s_full_index_set.intersection(Ch_cluster_index_set)
    
    # Intialize empty sets corresponding to the two classical index sets.
    # These will be inferred as the row or column index in the biindices.
    classical_E_index_set = set()
    classical_s_index_set = set()
    classical_removed_check_index_set = set()
    
    # Initialize a set of indices to represent the predicted error vector with this syndrome (for this cluster).
    predicted_cluster_error_vector_index_set = set()
    
    # There are two cases depending on whether this cluster corresponds to a set of horizontal or vertical qubits.
    # These may be inferred based on the value of an arbitrary bit taken from the cluster.
    for arbitrary_index in E_cluster_index_set:
        break
        
    # Since the horizontal qubits are indexed first, whether these qubits are horizontal or vertical can be inferred.
    if (arbitrary_index < HGP_code.num_h_qubits):
        # Horizontal Case
        for qubit_index in E_cluster_index_set:
            qubit_biindex = index_to_biindex(qubit_index,HGP_code.n2)
            # Since these horizontal qubits are in the same row, the classical index uses the column index.
            classical_E_index_set.add(qubit_biindex[1])
        # Likewise, infer the classical check indices; these also use the colum component of the check biindex.
        for s_index in s_cluster_index_set:
            s_biindex = index_to_biindex(s_index,HGP_code.r2)
            classical_s_index_set.add(s_biindex[1])
        # Finally, infer the biindex of removed checks, also using the column component.
        for check_index in removed_check_indices:
            check_biindex = index_to_biindex(check_index,HGP_code.r2)
            classical_removed_check_index_set.add(check_biindex[1])
    else:
        # Vertical Case
        for qubit_index in E_cluster_index_set:
            qubit_biindex = index_to_biindex(qubit_index,HGP_code.r2,HGP_code.num_h_qubits)
            # Since these vertical qubits are in the same column, the classical index uses the row index.
            classical_E_index_set.add(qubit_biindex[0])
        # Likewise, infer the classical check indices; these also use the row component of the check biindex.
        for s_index in s_cluster_index_set:
            s_biindex = index_to_biindex(s_index,HGP_code.r2)
            classical_s_index_set.add(s_biindex[0])
        # Finally, infer the biindex of removed checks, also using the column component.
        for check_index in removed_check_indices:
            check_biindex = index_to_biindex(check_index,HGP_code.r2)
            classical_removed_check_index_set.add(check_biindex[0])
            
    
    # CONFIRM FUNCTION IMPORT FROM HERE
    
    # DEBUG
    """
    if (arbitrary_index < HGP_code.num_h_qubits):
        print("Cluster corresponds to a horizontal classical stopping set.")
        print("   Arbitrary selected stopping set bit index:",arbitrary_index,
              "; biindex:",index_to_biindex(arbitrary_index,HGP_code.n2))
    else:
        print("Cluster corresponds to a vertical classical stopping set.")
        print("   Arbitrary selected stopping set bit index:",arbitrary_index,
              "; biindex:",index_to_biindex(arbitrary_index,HGP_code.r2,HGP_code.num_h_qubits))
    print("   Classical E index set:",classical_E_index_set)
    print("   Classical s index set:",classical_s_index_set)
    """

    # Next, the check indices need to be converted into the appropriate syndrome vector format.
    # This will allow it to be used with the classical syndrome analysis function.
    # Again, there are two cases depending on whether we use horizontal or vertical qubits.
    if (arbitrary_index < HGP_code.num_h_qubits):
        # Horizontal Case
        # Intialize this as a zero vector, and then set the nonzero values.
        classical_s = np.zeros(HGP_code.r2,dtype=int)
        for s_index in classical_s_index_set:
            classical_s[s_index] = 1
    else:
        # Vertical Case
        # Intialize this as a zero vector, and then set the nonzero values.
        classical_s = np.zeros(HGP_code.n1,dtype=int)
        for s_index in classical_s_index_set:
            classical_s[s_index] = 1

    # A solution may or may not exist; this is tracked with a boolean variable "solution_exists"
    
    # The predicted classical error value can now be inferred from the classical syndrome analysis function.
    # The parity check matrix used depends on whether these checks came from horizontal or vertical qubits.
    # NOTE: the horizontal case uses H2 (based on the geometric structure of the HGP code)
    if (arbitrary_index < HGP_code.num_h_qubits):
        # Horizontal case: use H2 as the parity check matrix.
        # Modified to zero the rows corresponding to removed checks.
        classical_H = HGP_code.H2.copy()
        for check_index in classical_removed_check_index_set:
            # Set every entry in this row of the matrix to zero.
            classical_H[check_index] = 0
            
        try:
            predicted_classical_e = perform_classical_syndrome_analysis_with_erasure(classical_H,
                                                                                     classical_s,
                                                                                     classical_E_index_set)
            solution_exists = True
        except:
            solution_exists = False
    else:
        # Vertical case: use H1 transpose as the parity check matrix. (CONFIRM THAT THIS IS CORRECT? - TEST WITH MATRIX SHAPE)
        # NOTE: the vertical case needs to use the transpose of H1? (can confrim that dimensions don't match otherwise)
        # Modified to zero the rows corresponding to removed checks.
        classical_H = HGP_code.H1.T.copy()
        for check_index in classical_removed_check_index_set:
            # Set every entry in this row of the matrix to zero.
            classical_H[check_index] = 0
            
        try:
            predicted_classical_e = perform_classical_syndrome_analysis_with_erasure(classical_H,
                                                                                     classical_s,
                                                                                     classical_E_index_set)
            solution_exists = True
        except:
            solution_exists = False

    # If a solution does exist, convert this back into an index set.
    if solution_exists:
        # Convert this predicted classical error into an index set, first by using biindices.
        predicted_classical_e_biindex_set = set()
        predicted_classical_e_index_set = set()

        # Again, there are two cases to consider depending on if this set corresponds to horizontal or vertical qubits.
        if (arbitrary_index < HGP_code.num_h_qubits):
            # Horizontal Case
            # Infer the one component of the biindex from the original arbitrary index.
            arbitrary_biindex = index_to_biindex(arbitrary_index,HGP_code.n2)
            for classical_bit_index in range(len(predicted_classical_e)):
                if (predicted_classical_e[classical_bit_index] == 1):
                    predicted_classical_e_biindex_set.add((arbitrary_biindex[0],classical_bit_index))
            # Convert these back into qubit indices.
            for classical_bit_biindex in predicted_classical_e_biindex_set:
                predicted_classical_e_index_set.add(biindex_to_index(classical_bit_biindex,HGP_code.n2))
        else:
            # Vertical Case
            # Infer the one component of the biindex from the original arbitrary index.
            arbitrary_biindex = index_to_biindex(arbitrary_index,HGP_code.r2,HGP_code.num_h_qubits)
            for classical_bit_index in range(len(predicted_classical_e)):
                if (predicted_classical_e[classical_bit_index] == 1):
                    predicted_classical_e_biindex_set.add((classical_bit_index,arbitrary_biindex[1]))
            # Convert these back into qubit indices.
            for classical_bit_biindex in predicted_classical_e_biindex_set:
                predicted_classical_e_index_set.add(biindex_to_index(classical_bit_biindex,HGP_code.r2,HGP_code.num_h_qubits))

        # The predicted_classical_e_index_set corresponds to the "lift" of the classical predicted error vector to the HGP code.
        # This predicted error vector corresponds to one of the classical stopping set connected components.
        # Note that the error vector from each component is neccessarily disjoint from the others.
        # Add it to the "final" classical error index set, which account for all classical predicted errors.
        predicted_cluster_error_vector_index_set.update(predicted_classical_e_index_set)
        
        # Update the set of erased qubit indices and triggered check indices to exclude those used in this classical correction.
        #final_E_index_set.difference_update(current_E_index_set)
        #final_s_index_set.difference_update(current_s_index_set)

        # DEBUG
        #print("Predicted cluster error vector index set:", predicted_cluster_error_vector_index_set)
        
        if (HGP_code.Hz_syn_index_set_for_X_err(predicted_cluster_error_vector_index_set) != s_cluster_index_set):
            raise Exception('The predicted error from correcting classical stopping sets does not yield the original syndrome!')

    return solution_exists, predicted_cluster_error_vector_index_set



# Recursive function to solve a tree of erased clusters.
# Clusters in this tree may share checks, and the classical solution on a cluster could change because of this.
# This function attempts to resolve these shared checks without a brute-force search checking all solutions.
# Dangling clusters are "peeled" from the tree so that no more than a single shared check is dealt with at a time.
# For a dangling cluster, we check whether a (classical) solution exists for both possible values (0 or 1) of the shared check.
# If a unique solution exists, then this forces the value of the shared syndrome bit on the adjacent cluster.
# If two solutions are possible, then a solution exists regardless of the value of check from the adjacent cluster.
# In this case, the adjacent cluster can be solved using the simpler classical method with the shared check removed.
# After finding a solution for the simpler problem, this forces a value of the removed check bit.
# This forced value determine which of the two solutions to use in earlier check.

# This function uses recursion to identify those shared checks for which two solutions are possible without fixing a choice.
# By passing to the recursive step, the simpler problem is considered for the adjacent cluster.
# When a solution is determined for the adjacent cluster, this is used to infer the choice after the recursion.

# The clusters can be ordered based on how to peel them from the cluster tree.
# We start by "popping" the first cluster from this ordered list and looking for solutions.
# The recursion stops when this list of clusters is exhausted.

# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# list_of_E_cluster_index_sets: the list of sets of erased qubit indices in a single cluster of a connected component.
# list_of_shared_checks_per_cluster: a list of sets of indices for checks that this cluster shares with other clusters.
# list_of_clusters_per_check: a list of indices of clusters adjacent to a given check.
# peelable_cluster_index_list: a list of indices for clusters which can be peeled; ordered based on order of peeling.
# set_of_two_solution_shared_check_indices: the set of shared check indices for which two solutions exist at a shallower level*
# s_comp_index_set: the set of indices for triggered syndrome bits in this connected component.

# Function Outputs:
# e_comp_pred_index_set: the index set of the predicted error vector on this component (combining all deeper recursive levels)

def solve_cluster_tree_by_recursive_peeling(HGP_code,
                                            list_of_E_cluster_index_sets_input,
                                            list_of_shared_checks_per_cluster_input,
                                            list_of_clusters_per_check_input,
                                            peelable_cluster_index_list_input,
                                            set_of_two_solution_shared_check_indices_input,
                                            s_comp_index_set):
    
    # Initialize sets for the outputs that will be returned.
    e_comp_pred_index_set = set()
    
    # Make copies of some sets that could be be modified; this should prevent some problems with overwriting data.
    list_of_E_cluster_index_sets = list_of_E_cluster_index_sets_input.copy()
    list_of_shared_checks_per_cluster = list_of_shared_checks_per_cluster_input.copy()
    list_of_clusters_per_check = list_of_clusters_per_check_input.copy()
    peelable_cluster_index_list = peelable_cluster_index_list_input.copy()
    set_of_two_solution_shared_check_indices = set_of_two_solution_shared_check_indices_input.copy()
    s_comp_updated_index_set = s_comp_index_set.copy()
    
    
    # DEBUG
    #print("Current peelable cluster index list:",peelable_cluster_index_list)
    
    
    # Before proceeding, check that the list of peelable clusters has not yet been exhausted.
    # If it has been exhausted, skip everything else; this terminates the recursive loop.
    if (len(peelable_cluster_index_list) > 0):
        # If there still exist clusters to be peeled, pop the first index from the list to analyze.
        cluster_index = peelable_cluster_index_list.pop(0)
        cluster_E_index_set = list_of_E_cluster_index_sets[cluster_index]
        
        # Infer the set of checks adjacent to the qubits in this set
        cluster_Ch_index_set = compute_adjacent_check_indices(HGP_code,cluster_E_index_set)
        
        # Infer any previously identified shared checks with two solutions (from shallower levels of recursion)
        # Any checks in this list must be zero-ed to solve a simpler system.
        local_two_solution_shared_check_indices = set_of_two_solution_shared_check_indices.intersection(cluster_Ch_index_set)
        
        
        # Since this tree is peelable, with the clusters ordered by peeling, shared checks are no longer shared after peeling.
        # Hence, each cluster should have at most one shared check by the time we encounter it.
        # The last cluster peeled should have no shared checks and other clusters will have exactly one.
        # Initialize some variables to track these possibilities.
        if (len(list_of_shared_checks_per_cluster[cluster_index]) == 0):
            last_cluster = True
            # Set a placeholder value for this non-existing shared check
            shared_check_index = -1     
        elif (len(list_of_shared_checks_per_cluster[cluster_index]) == 1):
            last_cluster = False
            # Infer the value of the unique shared check index by exploiting a loop with only a single pass through the set.
            for check_index in list_of_shared_checks_per_cluster[cluster_index]:
                shared_check_index = check_index
        elif (len(list_of_shared_checks_per_cluster[cluster_index]) > 1):
            # If this cluster has more than one shared check index, raise an exception.
            # This should never happen if the recursion is used correctly.
            raise Exception('Encountered problem with solving classical cluster tree; cannot peel a non-dangling cluster!')
            
        # The shared check now having been accounted for, we remove it from the list of shared checks per cluster.
        for local_cluster_index in list_of_clusters_per_check[shared_check_index]:
            # Fix the cluster adjacent to this check, and then remove this check from the list of those that are shared.
            list_of_shared_checks_per_cluster[local_cluster_index].discard(shared_check_index)
        
        
        # After identifying the shared check index, we must check whether one or two solutions exist based on this syndrome bit.
        # That is, we must test two cases:
        #     Case 1: A classical solution exists for this cluster using the current syndrome index set.
        #     Case 2: A classical solution exists for this cluster using the syndrome index set with the shared check flipped.
        # At least one of these two cases must be true, but it is possible that both are true.
        
        
        # If this is the last cluster, than we can compute a solution and disregard anything more about shared checks.
        if last_cluster:
            cluster_solution_exists, e_comp_pred_index_set = cluster_syndrome_analysis_with_zeroing(
                HGP_code,cluster_E_index_set,s_comp_index_set,local_two_solution_shared_check_indices)
        else:
            # If this is not the last cluster, then we must account for the possibility of two solutions.
            # We check both possibilities separately, keeping track of the predicted error vector if it does exist.
            # The two possibilities can be distinguished based on whether the shared syndrome bit is triggered or not.
            # For compactness refer to these two possibilities using:
            #    Case 1: "shared check tiggered" = _sct
            #    Case 2: "shared check not triggered" = _scnt
            cluster_solution_exists_sct, e_pred_cluster_index_set_sct = cluster_syndrome_analysis_with_zeroing(
                HGP_code,cluster_E_index_set,s_comp_index_set.union({shared_check_index}),
                local_two_solution_shared_check_indices)
            cluster_solution_exists_scnt, e_pred_cluster_index_set_scnt = cluster_syndrome_analysis_with_zeroing(
                HGP_code,cluster_E_index_set,s_comp_index_set.difference({shared_check_index}),
                local_two_solution_shared_check_indices)
            
            # If NO solution exists in either case, raise an exception; this should not be possible.
            if ((cluster_solution_exists_sct or cluster_solution_exists_scnt) == False):
                raise Exception('Encountered problem checking existence of two shared check solutions; neither exists!?')
            
            
            # To procede with the recursive step, we must determine whether one or two solutions exist.
            if (cluster_solution_exists_sct and cluster_solution_exists_scnt):
                # If a solution exists for both possibilities, we must add this check to the set of those with two solutions.
                set_of_two_solution_shared_check_indices.add(shared_check_index)
                
                # Pass now to the recursive call of the function; we need this before we can determine which solution to use.
                e_comp_pred_index_set_rec = solve_cluster_tree_by_recursive_peeling(
                    HGP_code,list_of_E_cluster_index_sets,list_of_shared_checks_per_cluster,list_of_clusters_per_check,
                    peelable_cluster_index_list,set_of_two_solution_shared_check_indices,s_comp_updated_index_set)
                
                # Which of the two solutions we use depends on the dot product of the error vector with the shared check row.
                # However, since we are working with sets rather than vectors, we determine this condition slightly differently.
                # The row corresponding to the shared check matches the set of qubit indices adjacent to this check.
                # This set of qubit indices is intersected with the predicted error vector index set obtained above.
                # If the number of entries is even, the dot product is 0 (mod 2); if odd, the dot product is 1 (mod 2).
                # These correspond to using the "scnt" and "sct" solutions, respectively.
                if (len(HGP_code.list_of_qubits_per_check[shared_check_index].intersection(e_comp_pred_index_set_rec))%2 == 0):
                    e_comp_pred_index_set = e_pred_cluster_index_set_scnt
                else:
                    e_comp_pred_index_set = e_pred_cluster_index_set_sct
                
                
            else:
                # Otherwise, only one of these two possibilities is true; we must use this possibility.
                # The exact effect on the next recursive call of the function depends on the original shared check bit.
                # If the original shared check bit is NOT TRIGGERED, then both clusters include/exclude this bit.
                # If the original shared check bit IS TIGGERED, the one cluster includes this bit, but the other does not.
                # Depending on the case, we may need to update the syndrome set for the next recursive pass.
                # Furthermore, specify the one existing solution as THE predicted error vector.
                
                if (shared_check_index not in s_comp_index_set):
                    # Here, the shared check is not triggered in the original syndrome.
                    # In this case, both clusters use the same value of the syndrome bit in computing their solutions.
                    # Determine the updated set depending on whether the solution using the triggered check exists or not.
                    if (cluster_solution_exists_sct):
                        s_comp_updated_index_set.add(shared_check_index)
                        e_comp_pred_index_set = e_pred_cluster_index_set_sct
                    else:
                        s_comp_updated_index_set.discard(shared_check_index)
                        e_comp_pred_index_set = e_pred_cluster_index_set_scnt
                else:
                    # Here, there shared check is triggered in the original syndrome.
                    # In this case, clusters use opposite values of their syndrome bit to avoid undoing each other.
                    # Again determine the updated syndrome set.
                    if (cluster_solution_exists_sct):
                        s_comp_updated_index_set.discard(shared_check_index)
                        e_comp_pred_index_set = e_pred_cluster_index_set_sct
                    else:
                        s_comp_updated_index_set.add(shared_check_index)
                        e_comp_pred_index_set = e_pred_cluster_index_set_scnt
                             
                # At this stage, we pass to the recursive step of the function (in the case where this cluster is not the last).
                # We use the updated syndrome index set in the recursive call of the function.
                e_comp_pred_index_set_rec = solve_cluster_tree_by_recursive_peeling(
                    HGP_code,list_of_E_cluster_index_sets,list_of_shared_checks_per_cluster,list_of_clusters_per_check,
                    peelable_cluster_index_list,set_of_two_solution_shared_check_indices,s_comp_updated_index_set)
            
            # The predicted error index set obtained in the recursive set can be combined with the existing predicted error.
            e_comp_pred_index_set.update(e_comp_pred_index_set_rec)
        
    # Return the predicted error vector and the updated sydrome triggered index set.
    return e_comp_pred_index_set



# Function consolidating all of the steps of clustering syndrome analysis for a given erasure pattern.
# This starts by identifying the connected components, and then clustering these components.
# For each component, a solution is found using the clustering method, and then these solutions are combined.

# This is a modified version of the original cluster decoder, which used a brute force approach to solving clusters.
# In this version, cycles of clusters are always ignored, and cluster trees are solved by an efficient peeling method.


# Function Inputs:
# HGP_code: the hypergraph product code class object being considered.
# E_index_set: the current set of erased qubit indices in the remaining erasure pattern.
# s_index_set: the current set of non-zero check indices so far, corresponding to the "triggered" syndrome bits.

# Function Outputs:
# predicted_e_index_set: the index set for the predicted error vector which gives the original syndrome.

def cluster_decoder(HGP_code,E_index_set,s_index_set):
    
    # Infer the set of adjacent check indices to the given set of erased qubit indices.
    Ch_index_set = compute_adjacent_check_indices(HGP_code,E_index_set)
    
    # Decompose the erasure set into a list of disjoint connected components.
    # The set of erased qubit indices and adjacent check indices are tracked separately.
    num_comps, list_of_E_comp_index_sets, list_of_Ch_comp_index_sets = decompose_subgraph_into_connected_components(
        HGP_code,E_index_set,Ch_index_set)
    
    # Initialize an empty predicted error vector index set; this vector will be returned at the end of the function.
    predicted_e_index_set = set()
    
    # Loop through each of these connected components and attempt to solve the restricted syndrome problem.
    for comp_index in range(num_comps):
        # Restrict to the qubits and checks corresponding to this connected component.
        comp_E_index_set = list_of_E_comp_index_sets[comp_index]
        comp_Ch_index_set = list_of_Ch_comp_index_sets[comp_index]
        
        # Infer the syndrome corresponding to the restriction to this particular connected component.
        comp_s_index_set = s_index_set.intersection(comp_Ch_index_set)
        
        # Decompose the current connected component into clusters.
        num_clusters, list_of_E_cluster_index_sets, list_of_Ch_cluster_index_sets = decompose_component_into_clusters(
            HGP_code,comp_E_index_set,comp_Ch_index_set)
        
        # Infer the checks which are shared between clusters, and which clusters are adjacent to each check.
        list_of_shared_checks_per_cluster,list_of_clusters_per_check = compute_list_of_shared_checks_per_cluster_and_vice_versa(
            HGP_code,list_of_E_cluster_index_sets)
        
        # This cluster may correspond to a tree or a cycle; determine which.
        cluster_is_peelable, peelable_cluster_index_list = is_cluster_decomposition_peelable(
            HGP_code,list_of_E_cluster_index_sets)
        
        # DEBUG
        #print("Cluster is peelable:",cluster_is_peelable,"; peelable cluster index list:",peelable_cluster_index_list)
        
        # This version of the cluster decoder cannot solve cycles of clusters, which occur when a decomposition is not a tree.
        # If this happens, raise an exception.
        if (cluster_is_peelable == False):
            raise Exception('Cycle of clusters detected; automatic decoder failure.')
            
        # Since clusters may share check bits, the solutions from two clusters may affect the same syndrome bit.
        # This fact must be taken into account when choosing a cluster solution which lifts to a full solution.
        # There are several nuanced cases to consider, and these are all handled by the following function.
        # This function uses the efficient peeling method to solve a tree of clusters recursively.
        pred_comp_e_index_set = solve_cluster_tree_by_recursive_peeling(
            HGP_code, list_of_E_cluster_index_sets, list_of_shared_checks_per_cluster, list_of_clusters_per_check,
            peelable_cluster_index_list,set(),comp_s_index_set)
            
        # Add the solution on this connected component to the total predicted error vector.
        predicted_e_index_set.update(pred_comp_e_index_set)
        
    
    # Raise an error if the predicted error vector does not match the original syndrome.
    if (HGP_code.Hz_syn_index_set_for_X_err(predicted_e_index_set) != s_index_set):
        raise Exception('The predicted error vector does not yield the original syndrome.')
    
    # Return the predicted error vector.
    return predicted_e_index_set



# THIS FUNCTION IS MODIFIED FROM ONE ASSOCIATED TO THE HGP_Code CLASS OBJECT
# The original function does not work as intended when applied to a local copy of the HGP_Code object (because the generators are not updated for erased qubits.)

# Function to determine whether the product of two overlapping generators is fully erased or not.
# Although written with the intention of applying to a pair of overlapping generators, this function should generalize as written.
# That is, given any set of generator indices, this function applies to the tensor-product of all of them.

def is_product_of_generators_fully_erased(list_of_generators,index_set):
        # The index_set should be a set object containing two (or more) indices, but a set being unordered prevents easy access.
        # Loop through this set to create a list of these generator objects by accessing their indices from the index set.
        # Because axiom of choice.
        gen_prod_list = []
        for index in index_set:
            gen_prod_list.append(list_of_generators[index])
        
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



# Function to combine the peeling and cluster decoders.

# The tries to solve error with the following steps:
#    1. peeling decoder (M=0)
#    2. peeling decoder (M=1)
#    3. peeling decoder (M=2)
#    4. peeling decoder (M=2) and cluster decoder

# Function Inputs:
# HGP_code: the class object representing the HGP code under consideration, with all of its parameters.
# E_index_set_input: the set of erased qubit indices in the erasure pattern
# s_index_set_input: the set of nonzero syndrome vector indices for an X-Pauli error supported on the erasure.

# Function Output:
# predicted_e_index_set: the set of nonzero indices of a predicted error vector with the given syndrome.
# combined_decoder_results_dict: a decoder consolidating the results, if successful (number of dangling checks used, etc.)


def combined_peeling_and_cluster_decoder(HGP_code,E_index_set_input,s_index_set_input):
    
    # Initialize some local versions of the input and the predicted error vector to return.
    predicted_e_index_set = set()
    E_index_set = E_index_set_input.copy()
    s_index_set = s_index_set_input.copy()
    
    # Copy the list of the check and generator objects from within the HGP_code class object.
    # This copy will allow us to update those checks/generators adjacent to erased qubits without over-writing the defaults.
    # Note that doing this requires creating a DEEPcopy (otherwise the objects themselves have the same memory location).
    list_of_checks = copy.deepcopy(HGP_code.list_of_checks)
    list_of_generators = copy.deepcopy(HGP_code.list_of_generators)
    
    
    # Initialize an empty set of dangling checks; a check is dangling if it satisfies: h=1 and v=0 OR h=0 and v=1
    set_of_dangling_check_indices = set()

    # Initialize a set of "entirely erased generator" indices.
    # These are generators for which all adjacent qubits are erased.
    set_of_entirely_erased_generator_indices = set()

    # Initialize a set of "entirely erased product generator" index-pairs.
    # These are tensor-products of two generators (which is also a generator) for which all adjacent qubits are erased.
    set_of_entirely_erased_product_generator_index_pairs = set()
    
    
    # Loop over the set of erased qubit indices and check whether adjacent checks can be added to these sets.
    for qubit_index in E_index_set:
        for check_index in HGP_code.list_of_checks_per_qubit[qubit_index]:
            # Update these checks with the erased qubits.
            # Note that the set intersection/union operations within the class objects ensure the correct indices are used.
            check = list_of_checks[check_index]
            check.add_erased_qubits(E_index_set)
            
            # If this check is dangling or critical, add its index to the corresponding list.
            if check.is_dangling():
                set_of_dangling_check_indices.add(check_index)
                
    # Loop over the set of erased qubit indices and update adjacent generators.
    for qubit_index in E_index_set:
        for generator_index in HGP_code.list_of_generators_per_qubit[qubit_index]:
            # Update these generators with the erased qubits.
            # Again, the set intersection/union operations within the class objects ensure correct indices are used.
            generator = list_of_generators[generator_index]
            generator.add_erased_qubits(E_index_set)

            # If this generator is completely erased, add it to the set.
            if generator.is_entirely_erased_generator():
                set_of_entirely_erased_generator_indices.add(generator_index)

    # Now that erased qubits have been updated in checks and generators, loop through products of generators.
    # Go through each of these products to determine whether to add to the list of entirely erased product generators.
    for index_pair in HGP_code.set_of_pairs_of_overlapping_generator_indices:
        if is_product_of_generators_fully_erased(list_of_generators,index_pair):
            set_of_entirely_erased_product_generator_index_pairs.add(index_pair)
            
            
    # Initialize some internal variables to keep track of progress of the decoder.
    total_dangling_checks_corrected = 0
    entirely_erased_generators_used = 0
    entirely_erased_generator_products_used = 0
    classical_stopping_sets_used = 0
    
    # Finally, initialize a dictionary to be returned at the end with the solution.
    # This dictionary will catalogue the steps used if a successful decoding occurs.
    combined_decoder_results_dict = {}
            
    
    while (s_index_set != set()):
        
        if (set_of_dangling_check_indices != set()):
            # 1. Remove dangling checks (M=0)
            
            # If the set of dangling checks is nonempty, then pop one of these checks and correct the corresponding qubit.
            dangling_check_index = set_of_dangling_check_indices.pop()
            dangling_check = list_of_checks[dangling_check_index]
            
            # If this check is still dangling, then proceed to correct the adjacent qubit.
            # The if-statement below is a safe guard against the possibility that this check has ceased to be dangling.
            if (dangling_check.is_dangling()):
                # 1. Identify the adjacent qubit and remove it from the erasure.
                # 2. Update this qubit in the predicted error vector based on the syndrome value.
                # 3. If this qubit is flipped, then also flip the syndrome bit values of any checks adjacent to this qubit.
                # 4. Remove this qubit from the erasure.
                # 5. Update the any checks and generators adjacent to this removed qubit.
                
                # 1. The index of this qubit (whether horizontal or vertical) is the lone adjacent qubit.
                qubit_index = dangling_check.horizontal_erased_qubits.union(dangling_check.vertical_erased_qubits).pop()
                
                # DEBUG
                #print("-Adjacent to dangling check",dangling_check.index,"is qubit",qubit_index)
                
                # 2. Based on the syndrome value for this check, modify the estimated error bit corresponding to this qubit.
                # An error is only detected if this syndrome bit is 1; if it is 0, do not change the estimated error.
                # In this case, we must also flip the values of the syndrome bits for ALL checks adjacent to this flipped qubit.
                if (dangling_check_index in s_index_set):
                    # Flip the value of this qubit in the estimated error.
                    if (qubit_index in predicted_e_index_set):
                        predicted_e_index_set.discard(qubit_index)
                    else:
                        predicted_e_index_set.add(qubit_index)
                    
                    # 3. All checks adjacent to this qubit will have their corresponding syndrome bit flipped.
                    # The index of these syndrome bits matches the check index; perform these flips with a loop.
                    for check_index in HGP_code.list_of_checks_per_qubit[qubit_index]:
                        if (check_index in s_index_set):
                            s_index_set.discard(check_index)
                        else:
                            s_index_set.add(check_index)
                            
                # 4. Update the set of erased qubit indices to exclude this corrected qubit.
                E_index_set.discard(qubit_index)
                
                # 5. Loop through the subset of checks and generators adjacent to this qubit.
                
                # For each, update the erasure, and determine whether an updated check becomes dangling or critical.
                for check_index in HGP_code.list_of_checks_per_qubit[qubit_index]:
                    adjacent_check = list_of_checks[check_index]
                    adjacent_check.remove_erased_qubit(qubit_index)
                    
                    if adjacent_check.is_dangling():
                        set_of_dangling_check_indices.add(adjacent_check.index)
                    else:
                        set_of_dangling_check_indices.discard(adjacent_check.index)
                        
                for generator_index in HGP_code.list_of_generators_per_qubit[qubit_index]:
                    adjacent_generator = list_of_generators[generator_index]
                    adjacent_generator.remove_erased_qubit(qubit_index)

                    if adjacent_generator.is_entirely_erased_generator():
                        set_of_entirely_erased_generator_indices.add(generator_index)
                    else:
                        set_of_entirely_erased_generator_indices.discard(generator_index)

                    # Loop through any generator-products involving this particular generator.
                    # If the product is entirely erased, add it to the list; otherwise, remove it.
                    for ol_gen_index in HGP_code.list_of_sets_of_overlapping_generator_indices_per_generator[generator_index]:
                        product_gen_index_pair = frozenset([generator_index,ol_gen_index])
                        if is_product_of_generators_fully_erased(list_of_generators,product_gen_index_pair):
                            set_of_entirely_erased_product_generator_index_pairs.add(product_gen_index_pair)
                        else:
                            set_of_entirely_erased_product_generator_index_pairs.discard(product_gen_index_pair)
                        
                # Increment the counter for the total number of dangling checks corrected.
                total_dangling_checks_corrected += 1
            
            
        elif (set_of_entirely_erased_generator_indices != set()):
            # 2. Remove erased generators (M=1)
            
            # Pop an index from this set of generator indices and select the corresponding generator.
            entirely_erased_generator_index = set_of_entirely_erased_generator_indices.pop()
            entirely_erased_generator = list_of_generators[entirely_erased_generator_index]

            # Choose a random erased qubit adjacent to this generator and remove it from the erasure.
            # In principal, any value on this qubit will do since this merely forces a value on the remaining adjacent qubits.
            # Pop this qubit from a copy of the union of the adjacent qubit indices (to avoid possibly modifying the generator.)
            qubit_index_to_remove_from_erasure = entirely_erased_generator.horizontal_erased_qubits.union(
                entirely_erased_generator.vertical_erased_qubits).copy().pop()

            # Remove this qubit from the erasure, then update any adjacent checks and generators.
            E_index_set.discard(qubit_index_to_remove_from_erasure)
            
            # For each, update the erasure, and determine whether an updated check becomes dangling or critical.
            for check_index in HGP_code.list_of_checks_per_qubit[qubit_index_to_remove_from_erasure]:
                adjacent_check = list_of_checks[check_index]
                adjacent_check.remove_erased_qubit(qubit_index_to_remove_from_erasure)
                
                if adjacent_check.is_dangling():
                    set_of_dangling_check_indices.add(adjacent_check.index)
                else:
                    set_of_dangling_check_indices.discard(adjacent_check.index)
                    
            for generator_index in HGP_code.list_of_generators_per_qubit[qubit_index_to_remove_from_erasure]:
                adjacent_generator = list_of_generators[generator_index]
                adjacent_generator.remove_erased_qubit(qubit_index_to_remove_from_erasure)

                if adjacent_generator.is_entirely_erased_generator():
                    set_of_entirely_erased_generator_indices.add(generator_index)
                else:
                    set_of_entirely_erased_generator_indices.discard(generator_index)

                # Loop through any generator-products involving this particular generator.
                # If the product is entirely erased, add it to the list; otherwise, remove it.
                for ol_gen_index in HGP_code.list_of_sets_of_overlapping_generator_indices_per_generator[generator_index]:
                    product_gen_index_pair = frozenset([generator_index,ol_gen_index])
                    if is_product_of_generators_fully_erased(list_of_generators,product_gen_index_pair):
                        set_of_entirely_erased_product_generator_index_pairs.add(product_gen_index_pair)
                    else:
                        set_of_entirely_erased_product_generator_index_pairs.discard(product_gen_index_pair)

            # At this stage, an "entirely erased" generator has had one adjacent qubit removed the erasure.
            # The hope is that this opens up new dangling checks, and we continue with the search.
            entirely_erased_generators_used += 1
            
            
        elif (set_of_entirely_erased_product_generator_index_pairs != set()):
            # 3. Remove erased products of generators (M=2)
            
            # Pop a pair of indices for the two generators in the product from this list.
            erased_product_generator_index_pair = set_of_entirely_erased_product_generator_index_pairs.pop()

            # As a safegaurd, confirm that this product generator is fully erased before proceeding.
            #if HGP_code.is_product_of_generators_fully_erased(erased_product_generator_index_pair):
            if is_product_of_generators_fully_erased(list_of_generators,erased_product_generator_index_pair):
                # Initilize a set to represent the erased qubits in this product generator.
                product_generator_qubit_indices = set()

                # Infer the set of indices of qubits in this product generator and populate this set.
                # This is done with a loop that should generalize to products of more than two generators, if ever needed.
                for local_gen_index in erased_product_generator_index_pair:
                    local_generator = HGP_code.list_of_generators[local_gen_index]
                    local_generator_qubit_indices_set = local_generator.horizontal_qubits.union(
                        local_generator.vertical_qubits).copy()
                    product_generator_qubit_indices.symmetric_difference_update(local_generator_qubit_indices_set)

                # Pop one of these qubit indices at random from the set, and remove this qubit from the erasure.
                qubit_index_to_remove_from_erasure = product_generator_qubit_indices.pop()

                # Remove this qubit from the erasure, then update any adjacent checks and generators.
                E_index_set.discard(qubit_index_to_remove_from_erasure)
                
                # For each, update the erasure, and determine whether an updated check becomes dangling or critical.
                for check_index in HGP_code.list_of_checks_per_qubit[qubit_index_to_remove_from_erasure]:
                    adjacent_check = list_of_checks[check_index]
                    adjacent_check.remove_erased_qubit(qubit_index_to_remove_from_erasure)
                    
                    if adjacent_check.is_dangling():
                        set_of_dangling_check_indices.add(adjacent_check.index)
                    else:
                        set_of_dangling_check_indices.discard(adjacent_check.index)
                        
                for generator_index in HGP_code.list_of_generators_per_qubit[qubit_index_to_remove_from_erasure]:
                    adjacent_generator = list_of_generators[generator_index]
                    adjacent_generator.remove_erased_qubit(qubit_index_to_remove_from_erasure)

                    if adjacent_generator.is_entirely_erased_generator():
                        set_of_entirely_erased_generator_indices.add(generator_index)
                    else:
                        set_of_entirely_erased_generator_indices.discard(generator_index)

                    # Loop through any generator-products involving this particular generator.
                    # If the product is entirely erased, add it to the list; otherwise, remove it.
                    for ol_gen_index in HGP_code.list_of_sets_of_overlapping_generator_indices_per_generator[generator_index]:
                        product_gen_index_pair = frozenset([generator_index,ol_gen_index])
                        if is_product_of_generators_fully_erased(list_of_generators,product_gen_index_pair):
                            set_of_entirely_erased_product_generator_index_pairs.add(product_gen_index_pair)
                        else:
                            set_of_entirely_erased_product_generator_index_pairs.discard(product_gen_index_pair)

                # At this stage, an "entirely erased product of generators" has had one adjacent qubit removed the erasure.
                # The hope is that this opens up new dangling checks, and we continue with the search.
                entirely_erased_generator_products_used += 1
            
        else:
            # 4. Apply the cluster decoder to the remaining erasure pattern/syndrome.
            
            # The cluster decoder is the last method we apply; if it fails, then flag this as a decoding failure.
            try:
                cluster_decoder_pred_e_index_set = cluster_decoder(HGP_code,E_index_set,s_index_set)
                
                # Confirm that the predicted error vector obtained in this way matches the remaining syndrome.
                if (HGP_code.Hz_syn_index_set_for_X_err(cluster_decoder_pred_e_index_set) != s_index_set):
                    raise Exception('Decoding failure: cluster decoder predicted error vector does not give correct syndrome.')
                else:
                    # If the syndromes do match, then this a decoding success.
                    # We may update the total predicted error vector accordingly.
                    # Also, we may replace the E_index_set and s_index_set with empty sets.
                    predicted_e_index_set.update(cluster_decoder_pred_e_index_set)
                    E_index_set = set()
                    s_index_set = set()
                    
                    # Increment the counter for classical stopping sets (those sets not correctable by peeling/generators alone)
                    classical_stopping_sets_used += 1
                
            except:
                raise Exception('Decoding failure: exhausted dangling checks/erased generators and cluster decoder failed.')
                
    
    # Update the dictonary of results.
    combined_decoder_results_dict['total_dangling_checks_corrected'] = total_dangling_checks_corrected
    combined_decoder_results_dict['entirely_erased_generators_used'] = entirely_erased_generators_used
    combined_decoder_results_dict['entirely_erased_generator_products_used'] = entirely_erased_generator_products_used
    combined_decoder_results_dict['classical_stopping_sets_used'] = classical_stopping_sets_used
    
    
    # Return the predicted error vector and the dictionary of results.
    return predicted_e_index_set, combined_decoder_results_dict



# Function to run a simulation of the combined peeling/cluster decoder a specificed number of times at fixed erasure rate.
# We track the success of the decoding based on how many techniques were required to recover a predicted error vector.
# We distinguish between the peeling decoder (M = 0, 1, or 2), and this combined with the cluster decoder.
# Note that a decoding success requiring fewer techniques is also considered a success for the more complicated decoder.
# Tracking the successes this way allows all four variants of the decoder to be simulated simultaneously.
# (This is quite helpful and efficient for generating data that require longer simulations!)


# Function Inputs:
# HGP_code: the hypergraph product code class object to use with the decoding algorithm.
# num_iterations: the number of tests to run
# erasure_rate: the erasure rate to use with this decoder.

# Function Outputs:
# simulation_results_dict: a dictionary storing the results of the simulation (number of failures, successes, etc.)

def combined_peeling_cluster_decoder_simulation(HGP_code,num_iterations,erasure_rate):
    
    # Initialize some variables to track the decoder's performance.
    num_successes_peeling_M0 = 0
    num_successes_peeling_M1 = 0
    num_successes_peeling_M2 = 0
    num_successes_peeling_M2_cluster = 0
    
    # We make a distinction between "true" decoder failures (where the decoder cannot complete) and logical errors.
    # However, these are both counted as decoder failures when computing the failure rate, and hence combined at the end.
    num_true_decoding_failures = 0
    num_non_trivial_logical_errors = 0
    
    # Rather than return the individual variables, consolidate these into a dictionary and return this at the end.
    simulation_results_dict = {}

    # Based on the total number of iterations, write a print statement to keep track of the progress.
    # My default, we will make such statements in intervals of 20% of the total number of simulations.
    progress_count = int(num_iterations/2)
    
    # Repeat the simulation the number of times specified.
    for iterations in range(num_iterations):
        # For each new iteration, generate a new random erasure pattern E and error vector e, and compute the new syndrome.
        # In the erasure-decoder, we always use an error rate of 0.5.
        E_index_set, e_index_set = generate_random_erasure_and_error_index_sets(HGP_code.num_qubits,erasure_rate,0.5)
        s_index_set = HGP_code.Hz_syn_index_set_for_X_err(e_index_set)
        
        # Attempt the to correct the error using the following four modified versions of the decoder:
        # 1. Peeling decoder only (M=0)
        # 2. Peeling decoder and erased generators (M=1)
        # 3. Peeling decoder, erased generators, and erased products of generators (M=2)
        # 4. Peeling decoder (M=2) and then cluster decoder (using only cluster trees)
        try:
            # Run the decoder using the given erasure pattern and syndrome.
            # We may infer the rseults from the returned dictionary.
            predicted_e_index_set, combined_decoder_results_dict = combined_peeling_and_cluster_decoder(
                HGP_code,E_index_set,s_index_set)
            
            # The total error is the symmetric difference of the original and predicted errors.
            total_e_index_set = e_index_set.symmetric_difference(predicted_e_index_set)
            
            # DEBUG
            #print("Original error, predicted, and total:", e_index_set, predicted_e_index_set, total_e_index_set)
            #print("Combined results dictionary:",combined_decoder_results_dict)
            
            # First, check whether the total error vector is a non-trivial logical error or not.
            # If not, we infer what kind of success this should be counted as.
            if HGP_code.is_non_trivial_X_logical_error_index_set(total_e_index_set):
                num_non_trivial_logical_errors += 1
            else:
                if (combined_decoder_results_dict['classical_stopping_sets_used'] > 0):
                    # If at least one classical stopping set was used, this counts as (4).
                    num_successes_peeling_M2_cluster += 1
                elif (combined_decoder_results_dict['entirely_erased_generator_products_used'] > 0):
                    # If no classical stopping sets, but at least one erased product of generators was used,
                    # this counts as both (4) and (3).
                    num_successes_peeling_M2_cluster += 1
                    num_successes_peeling_M2 += 1
                elif (combined_decoder_results_dict['entirely_erased_generators_used'] > 0):
                    # If neither classical stopping sets nor erased products of generators were used, but at least one
                    # erased generator was used, this counts as (4), (3), and (2).
                    num_successes_peeling_M2_cluster += 1
                    num_successes_peeling_M2 += 1
                    num_successes_peeling_M1 += 1
                else:
                    # If none of these other things were used, but the decoder was successful anyway,
                    # then this counts as a success for all four variants of the decoder.
                    # Notably, this is true even if no dangling checks were used (for example, when the syndrome = 0).
                    num_successes_peeling_M2_cluster += 1
                    num_successes_peeling_M2 += 1
                    num_successes_peeling_M1 += 1
                    num_successes_peeling_M0 += 1
                    
        except:
            # If the decoder failed to predict an error, this is a true decoding failure.
            num_true_decoding_failures += 1

        # Write a print statment to track the progress of the simulation
        if ((iterations+1)%progress_count == 0):
            print("Current progress: ",iterations+1," simulations completed at ",erasure_rate," erasure rate.")
            
    # After finishing the simulation, populate the results dictionary using these variables.
    simulation_results_dict['total_trials'] = num_iterations
    simulation_results_dict['erasure_rate'] = erasure_rate
    simulation_results_dict['num_true_decoder_failures'] = num_true_decoding_failures
    simulation_results_dict['num_non_trivial_logical_errors'] = num_non_trivial_logical_errors
    simulation_results_dict['num_total_failures'] = num_true_decoding_failures + num_non_trivial_logical_errors

    # Record the total number of successes per type of decoder.
    # These are already combined where needed in the above else/if statements.
    simulation_results_dict['num_total_successes_peeling_M0'] = num_successes_peeling_M0
    simulation_results_dict['num_total_successes_peeling_M1'] = num_successes_peeling_M1
    simulation_results_dict['num_total_successes_peeling_M2'] = num_successes_peeling_M2
    simulation_results_dict['num_total_successes_peeling_M2_cluster'] = num_successes_peeling_M2_cluster
    
    # Finally, compute the failure rates corresponding to these.
    simulation_results_dict['failure_rate_peeling_M0'] = ((num_iterations - num_successes_peeling_M0)/float(num_iterations))
    simulation_results_dict['failure_rate_peeling_M1'] = ((num_iterations - num_successes_peeling_M1)/float(num_iterations))
    simulation_results_dict['failure_rate_peeling_M2'] = ((num_iterations - num_successes_peeling_M2)/float(num_iterations))
    simulation_results_dict['failure_rate_peeling_M2_cluster'] = (
        (num_iterations - num_successes_peeling_M2_cluster)/float(num_iterations))
            
    # Return the number of decoding failures, logical errors, and decoding successes.
    return simulation_results_dict



# Function to run the combined peeling and cluster decoder while varying the erasure rate.
# This is used to create a dictionary of data that can then be plotted.

# Function Inputs:
# HGP_code: the hypergraph product code class object to use with the decoding algorithm.
# max_erasure_rate: the maximum erasure rate to test the decoder with.
# steps: the number of times of the erasure rate will be varied.
# num_iterations: the number of tests to run for each fixed erasure rate.
# min_erasure_rate: the minimum erasure rate to test the decoder with.

# Function Outputs:
# list_of_decoder_performance_dicts: a list of dictionaries tracking the decoder's performance at different error rates.

def run_combined_peeling_cluster_decoder_varying_erasure_rate(HGP_code,max_erasure_rate,steps,num_iterations,min_erasure_rate=0):
    
    # Initialize a dictionary to store the performance information for this code.
    list_of_decoder_performance_dicts = []
    
    # Compute the step size of the erasure variable based on the maximum erasure rate and the number of steps.
    step_size = (max_erasure_rate - min_erasure_rate)/float(steps)
    erasure_rate = min_erasure_rate
    
    # Test the performance of the dictionary in error rate steps of size step_size until reaching the maximum error rate.
    # For each fixed error_rate, record a dictionary tracking the performance.
    while (erasure_rate < max_erasure_rate):
        # Increment the erasure rate by the step size.
        erasure_rate += step_size
        
        # Run the simulation using this erasure rate; I will assume that cluster cycles are not excluded.
        simulation_results_dict = combined_peeling_cluster_decoder_simulation(HGP_code,num_iterations,erasure_rate)
        
        # Append this dictionary to the list
        list_of_decoder_performance_dicts.append(simulation_results_dict)
        
    # Return the list of performance dicts.
    return list_of_decoder_performance_dicts



    # Function to write the data from a list of performance dictionaries to a text file.
# The function generates a default file name base on the HGP code, but this can be overwritten.

# Function Inputs:
# HGP_code: the Hypergraph Product Code class object, used to infer some information for the default file name conventions.
# list_of_perf_dicts: a list of performance dictionaries storing information about the results of the decoder simulation.
# ArrayJob: a boolean value to indicate whether this simulation was launched alone or as part of a batch.
# file_name: a string that can be used to name file (WITHOUT the .txt extension); or can leave blank to use default name.

# Function Outputs:
# file_name: the string used to name of the file where this data is written.

def write_list_of_performance_dictionaries_to_file(HGP_code,list_of_perf_dicts,ArrayJob=False,file_name=''):
  
    # In the simpler case of an isolated simulation, we generate a single file.
    # If the user has provided a file name, use this.
    if (file_name == ''):
        # Otherwise, infer a default filename based on the parameters of the HGP code.
        HGP_file_name_base = "["+str(HGP_code.num_qubits)+","+str(HGP_code.dim)+"]_HGP_code"
        total_trials = str(list_of_perf_dicts[0]['total_trials'])

        file_name = HGP_file_name_base+"_peeling_cluster_decoder_performance_data_"+str(total_trials)+"_trials_"
        
        # If this is an array job, then we also include the index from the array in the file name.
        if (ArrayJob):
            file_name = file_name+"v"+str(sys.argv[1])+".txt"
        else:
            # Otherwise, we just attach today's date (which is useful for testing).
            file_name = file_name+str(date.today())+".txt"

        # DEBUG:
        #print("File name:", file_name)
    else:
        # If the user specified a file name, add the .txt extension to this.
        # If it is also an array job, then include the index as well.
        if (ArrayJob):
            file_name = file_name+"_v"+str(sys.argv[1])+".txt"
        else:
            file_name = file_name+".txt"
            
    # There are two cases, depending on whether this is an isolated simulation or an array job.
    # If it is an array job, then create an intermediate folder to store this file.
    if (ArrayJob):
        folder_name = "Peeling_Cluster_Decoder_Array_Job_Folder_"+HGP_file_name_base
        file_path = "./"+folder_name+"/"+file_name
    else:
        file_path = "./"+file_name
          
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path,'w') as data:
        data.write("[")
        for row_dict in list_of_perf_dicts:
            data.write(str(row_dict)+",\n")
        data.write("]")
    
    # Return the string specifying the file name.
    return file_name



def main():

    print("Hello, world!")
    
    # 3x3 toric code, used for running small tests.
    #Toric3_sample_performance_dictionary = run_combined_peeling_cluster_decoder_varying_erasure_rate(Toric3,0.32,16,100)
    #write_list_of_performance_dictionaries_to_file(Toric3,Toric3_sample_performance_dictionary,False,file_name='Toric3_sample_data')
    #print(Toric3_sample_performance_dictionary[-1])

    # fix a number of trials to run
    num_trials = 25000

    # 625 qubit PEG code
    C_625 = construct_HGP_code_from_classical_H_text_file('PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt')
    C_625_perf_dict_list = run_combined_peeling_cluster_decoder_varying_erasure_rate(C_625,0.32,16,num_trials)

    #write_list_of_performance_dictionaries_to_file(C_625,C_625_perf_dict_list,ArrayJob=False)
    write_list_of_performance_dictionaries_to_file(C_625,C_625_perf_dict_list,ArrayJob=True)

    # 1225 qubit PEG code
    C_1225 = construct_HGP_code_from_classical_H_text_file('PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt')
    C_1225_perf_dict_list = run_combined_peeling_cluster_decoder_varying_erasure_rate(C_1225,0.32,16,num_trials)
    write_list_of_performance_dictionaries_to_file(C_1225,C_1225_perf_dict_list,ArrayJob=True)

    # 1600 qubit PEG code
    C_1600 = construct_HGP_code_from_classical_H_text_file('PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt')
    C_1600_perf_dict_list = run_combined_peeling_cluster_decoder_varying_erasure_rate(C_1600,0.32,16,num_trials)
    write_list_of_performance_dictionaries_to_file(C_1600,C_1600_perf_dict_list,ArrayJob=True)

    # 2025 qubit PEG code
    C_2025 = construct_HGP_code_from_classical_H_text_file('PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt')
    C_2025_perf_dict_list = run_combined_peeling_cluster_decoder_varying_erasure_rate(C_2025,0.32,16,num_trials)
    write_list_of_performance_dictionaries_to_file(C_2025,C_2025_perf_dict_list,ArrayJob=True)


    

main()