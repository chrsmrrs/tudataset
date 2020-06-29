import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation

dataset = "PROTEINS"

# Download datasets.
dp.get_dataset(dataset)
# Output datasets as a list of graphs.
graph_db = tud_to_networkx(dataset)

