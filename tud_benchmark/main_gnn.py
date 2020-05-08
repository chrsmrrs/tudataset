import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN, GINE, GINWithJK, GINEWithJK, GIN0
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

import os.path as osp

def main():
    num_reps = 3

    # Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True],["NCI109", True], ["PROTEINS", True],["PTC_FM", True],
               ["REDDIT-BINARY", False], ["ENZYMES", True]]


    results = []
    for d, use_labels in dataset:
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GIN0, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN0 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN0 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

    ####################################################################################################################

    exit()

    # Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
               ["REDDIT-BINARY", False], ["ENZYMES", True]]

    results = []
    # for d, use_labels in dataset:
    #     dp.get_dataset(d)
    #
    #     acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
    #                                    start_lr=0.01, num_repetitions=num_reps, all_std=True)
    #     print(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
    #     results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
    #
    #     acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
    #                                    start_lr=0.01, num_repetitions=num_reps, all_std=True)
    #     print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
    #     results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

    ####################################################################################################################
    # Larger datasets using LIBLINEAR.
    dataset = [["MCF-7", True], ["MOLT-4", True]]

    for d, use_labels in dataset:
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GINEWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                                       batch_size=128, start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINEWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GINE, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))



    dataset = [["TRIANGLES", False],
               ["github_stargazers", False],
               ["reddit_threads", False]]

    for d, use_labels in dataset:
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                                       batch_size=128, start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

    print("DONE! :*")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
