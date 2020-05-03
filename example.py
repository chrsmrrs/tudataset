import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation

# Download datasets.
classes = dp.get_dataset("ENZYMES")

all_matrices = []
for i in range(1, 6):
    # Use node labels and no edge labels.
    gm = kb.compute_wl_1_dense("ENZYMES", i, True, False)
    # Apply cosine normalization.
    gm = aux.normalize_gram_matrix(gm)
    all_matrices.append(gm)

# Perform 10 repetions of 10-CV using LIBSVM.
print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))

