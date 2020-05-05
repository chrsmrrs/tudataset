import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN, GINE, GINWithJK, GINEWithJK


def main():
    # Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
               ["REDDIT-BINARY", False], ["ENZYMES", True]]

    results = []
    for d, use_labels in dataset:
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=10, all_std=True)
        print(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01, num_repetitions=10, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

    ####################################################################################################################
    # Larger datasets using LIBLINEAR.
    dataset = [["MCF-7", True, True], ["MOLT-4", True, True]]

    for d, use_labels in dataset:
        dp.get_dataset(dataset)

        acc, s_1, s_2 = gnn_evaluation(GINEWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                                       batch_size=128, start_lr=0.01,
                                       num_repetitions=10, all_std=True)
        print(d + " " + "GINEWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GINE, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01,
                                       num_repetitions=10, all_std=True)
        print(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))



    dataset = [["TRIANGLES", False, False],
               ["github_stargazers", False, False],
               ["reddit_threads", False, False]]

    for d, use_labels in dataset:
        dp.get_dataset(dataset)

        acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                                       batch_size=128, start_lr=0.01,
                                       num_repetitions=10, all_std=True)
        print(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=128,
                                       start_lr=0.01,
                                       num_repetitions=10, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

    print("DONE! :*")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
