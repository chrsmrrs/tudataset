import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GINWithJK


# TODO: Add one-hot.

def main():


    d = "ENZYMES"
    dp.get_dataset(d)

    results = gnn_evaluation(GINWithJK, d, [2,3,4,5], [64], max_num_epochs=100, batch_size=128, start_lr=0.01, num_repetitions=10, all_std=True)
    print(results)

    print(results)


if __name__ == "__main__":
    main()
