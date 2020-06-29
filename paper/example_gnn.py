import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN

dataset = "PROTEINS"
use_labels = True
dp.get_dataset(dataset)

print(gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                     batch_size=128, start_lr=0.01, num_repetitions=10, all_std=True))
