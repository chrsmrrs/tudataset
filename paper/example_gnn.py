import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN

dataset = "PROTEINS"
use_labels = True

# Download dataset.
dp.get_dataset(dataset)

# Optimize the number of layers ({1,2,3,4,5} and
# the number of hidden features ({32,64,128}),
# set the maximum nummber of epochs to 200,
# batch size to 64,
# starting learning rate to 0.01, and
# number of repetitions for 10-CV to 10.
print(gnn_evaluation(GIN, dataset, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                     batch_size=64, start_lr=0.01, num_repetitions=10, all_std=True))
