import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GINWithJK, GINEWithJK
import os.path as osp
from torch_geometric.datasets import TUDataset
# TODO: Add one-hot.




def main():

    # # Smaller datasets using LIBSVM.
    # dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
    #              ["REDDIT-BINARY", False], ["ENZYMES", True]]
    #
    # results = []
    #
    # for d, use_labels in dataset:
    #     dataset = d
    #     dp.get_dataset(dataset)
    #
    #     acc, s_1, s_2 = gnn_evaluation(GINWithJK, d, [1,2,3,4,5], [32,64,128], max_num_epochs=200, batch_size=128, start_lr=0.01, num_repetitions=10, all_std=True)
    #     print(d + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
    #     results.append(d + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))
    #
    # print("DONE! :*")
    # for r in results:
    #     print(r)

    # Smaller datasets using LIBSVM.
    dp.get_dataset("SF-295")

    acc, s_1, s_2 = gnn_evaluation(GINE, "SF-295", [3], [64], max_num_epochs=200, batch_size=128, start_lr=0.01, num_repetitions=10, all_std=True)
    print("SF-295" + " " + "WL1 " + str(acc) + " " + str(s_1) + " " + str(s_2))




if __name__ == "__main__":
    main()
