import numpy as np
from surprise import Dataset
from surprise import SVD
import pandas as pd
import pickle
import surprise
from surprise import Reader
from surprise import Dataset




if __name__ == '__main__':
    data_file = open("./data/toy_dataset.pickle", 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)

    ratings_dict = {'user': train_u,
                    'item': train_v,
                    'rate': train_r}
    df = pd.DataFrame(ratings_dict)
    test_ratings_dict = {'user': test_u,
                    'item': test_v,
                    'rate': test_r}
    test_df = pd.DataFrame(test_ratings_dict)

    reader = Reader(rating_scale=(0.5, 4.0))

    data = Dataset.load_from_df(df[['user', 'item', 'rate']], reader)
    test_data = Dataset.load_from_df(test_df[['user', 'item', 'rate']], reader)

    trainset = data.build_full_trainset()
    testset = test_data.build_full_trainset().build_testset()


    #######################################################################
    ## Use Probabalistic Matrix Factorization #############################
    #######################################################################e
    algo=SVD(biased=False)
    algo.fit(trainset)

    rmse = surprise.accuracy.rmse(algo.test(testset))
    mae = surprise.accuracy.mae(algo.test(testset))

    print("rmse: %.4f, mae:%.4f " % ( rmse, mae))
