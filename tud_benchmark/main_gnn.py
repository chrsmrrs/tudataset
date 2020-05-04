import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN


# TODO: Add one-hot.

def main():


    d = "PROTEINS"
    dp.get_dataset(d)

    results = gnn_evaluation(GIN, d, [2,3,5,6], [32,64,128], max_num_epochs=100, batch_size=128, start_lr=0.01, num_repetitions=1, all_std=True)
    print(results)

    print(results)


if __name__ == "__main__":
    main()
