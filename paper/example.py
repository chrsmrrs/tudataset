import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation

# Download dataset.
classes = dp.get_dataset("ENZYMES")
use_labels, use_edge_labels = True, False

all_matrices = []
# Compute 1-WL kernel for 1 to 5 iterations.
for i in range(1, 6):
    # Use node labels and no edge labels.
    gm = kb.compute_wl_1_dense("ENZYMES", i, use_labels, use_edge_labels)
    # Apply cosine normalization.
    gm = aux.normalize_gram_matrix(gm)
    all_matrices.append(gm)

# Perform 10 repetitions of 10-CV using LIBSVM.
print(kernel_svm_evaluation(all_matrices, classes,
                            num_repetitions=10, all_std=True))

