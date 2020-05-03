from __future__ import division

import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GINWithJK, GraphSAGEWithJK


# TODO: Add one-hot.

def main():
    datasets = [
        #["ENZYMES", True],
        # ["IMDB-BINARY", False], ["IMDB-MULTI", False],["REDDIT-BINARY", False]
        #["NCI1", True],
        ["NCI1", True]]

    d = "IMDB-BINARY"
    dp.get_dataset(d)

    results = gnn_evaluation(GINWithJK, d, [2,3,5,6], [32,64,128], max_num_epochs=100, batch_size=32, start_lr=0.001, num_repetitions=1, all_std=True)
    print(results)




    print(results)


if __name__ == "__main__":
    main()
