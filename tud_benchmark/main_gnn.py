import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN, GINE, GINEWithJK, GINWithJK


def main():
    num_reps = 10

    ### Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
               ["REDDIT-BINARY", False], ["ENZYMES", True]]

    results = []
    for d, use_labels in dataset:
        # Download dataset.
        dp.get_dataset(d)

        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # GIN with jumping knowledge, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
                                       batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINWithJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

    num_reps = 3
    print(num_reps)

    ### Midscale datasets.
    dataset = [["MOLT-4", True, True], ["Yeast", True, True], ["MCF-7", True, True]]

    for d, use_labels, _ in dataset:
        print(d)
        dp.get_dataset(d)

        # GINE (GIN with edge labels), dataset d, 3 layers, hidden dimension in {64}.
        acc, s_1, s_2 = gnn_evaluation(GINE, d, [3], [64], max_num_epochs=200,
                                       batch_size=64, start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # GINE (GIN with edge labels) with jumping knowledge, dataset d, 3 layers, hidden dimension in {64}.
        acc, s_1, s_2 = gnn_evaluation(GINEWithJK, d, [3], [64], max_num_epochs=200,
                                       batch_size=64,
                                       start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINEJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINEJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

    dataset = [["reddit_threads", False, False],
               ["github_stargazers", False, False],
               ]

    for d, use_labels, _ in dataset:
        print(d)
        dp.get_dataset(d)

        # GINE (GIN with edge labels), dataset d, 3 layers, hidden dimension in {64}.
        acc, s_1, s_2 = gnn_evaluation(GIN, d, [3], [64], max_num_epochs=200,
                                       batch_size=64, start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

        # GINE (GIN with edge labels) with jumping knowledge, dataset d, 3 layers, hidden dimension in {64}.
        acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [3], [64], max_num_epochs=200,
                                       batch_size=64,
                                       start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINJK " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINJK " + str(acc) + " " + str(s_1) + " " + str(s_2))

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
