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

    for d, use_labels in datasets:

        classes = dp.get_dataset(d)

        results = gnn_evaluation(GINWithJK, d, [2], [64], max_num_epochs=50, batch_size=32, start_lr=0.01,
                                 num_repetitions=1,
                                 all_std=True)
        print(results)
        exit()


        results = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [16, 32, 64, 128], max_num_epochs=200, batch_size=32, start_lr=0.01,
                                 num_repetitions=10,
                                 all_std=True)
        print(results)

        results = gnn_evaluation(GraphSAGEWithJK, d, [1, 2, 3, 4, 5], [16, 32, 64, 128], max_num_epochs=200,
                                 batch_size=32, start_lr=0.01, num_repetitions=10,
                                 all_std=True)
        print(results)

    print(results)


if __name__ == "__main__":
    main()
