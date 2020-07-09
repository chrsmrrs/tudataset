import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation

# Download dataset.
classes = dp.get_dataset("MOLT-4")
use_labels, use_edge_labels = True, True

all_matrices = []
# Compute 1-WL kernel for 1 to 5 iterations.
for i in range(1, 6):
    # Use node labels and edge labels.
    gm = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
    # Apply \ell_2 normalization.
    gm_n = aux.normalize_feature_vector(gm)
    all_matrices.append(gm_n)

# Perform 10 repetitions of 10-CV using LIBINEAR.
print(linear_svm_evaluation(all_matrices, classes,
                            num_repetitions=10, all_std=True))