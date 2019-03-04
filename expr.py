from sacred import Experiment
from script import set_seed, main, load_data, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import sacred
import pandas as pd
import torch
import numpy as np
import os

expr = sacred.Experiment('insincere')
mongo_url = r'mongodb://user:password@mongoserver.com:port'
mongo_db = 'expr_insincere'
ob = sacred.observers.MongoObserver.create(url=mongo_url, db_name=mongo_db)
expr.observers.append(ob)
expr.add_config('config.yaml')


@expr.automain
def expr_func(random_seed, epochs, fine_tuning_epochs, batch_size, dropout, learning_rate, learning_rate_max_offset,
              threshold,
              max_vocab_size, max_seq_len, embed_size, n_folds, share, n_repeat, debug):
    set_seed(random_seed)
    print('seed: {}'.format(random_seed))
    # TODO try 5*2 fold
    print_every_step = 500
    train_df, _ = load_data(debug=debug)
    train_df, test_df = train_test_split(train_df, test_size=0.02)
    targets_te = test_df['target']
    scores_te = []
    scores_va = []
    for n in range(n_repeat):
        if n_folds > 1:
            predictions_te, scores, thresholds,coeffs = cv(train_df, test_df, n_folds=n_folds, epochs=epochs,
                                                    batch_size=batch_size,
                                                    learning_rate=learning_rate, threshold=threshold,
                                                    max_vocab_size=max_vocab_size, embed_size=embed_size,
                                                    print_every_step=print_every_step, share=share, dropout=dropout,
                                                    learning_rate_max_offset=learning_rate_max_offset,
                                                    fine_tuning_epochs=fine_tuning_epochs, max_seq_len=max_seq_len)
            print(coeffs)
            threshold_e = np.array(thresholds).mean()
            best_score = -1
            best_threshold = None
            for t in np.arange(0, 1, 0.01):
                score = f1_score(targets_te, predictions_te > t)
                if score > best_score:
                    best_score = score
                    best_threshold = t
            print('best threshold on test set: {:.2f} score {:.4f}'.format(best_threshold, best_score))
            scores_te.append(
                [f1_score(targets_te, predictions_te > threshold_e), roc_auc_score(targets_te, predictions_te),
                 precision_score(targets_te, predictions_te > threshold_e),
                 recall_score(targets_te, predictions_te > threshold_e)])
            scores_va.extend(scores)
        else:
            train_df, valid_df = train_test_split(train_df, test_size=0.02)
            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            predictions_te, predictions_va, targets_va, best_threshold = main(train_df, valid_df, test_df,
                                                                              epochs=epochs,
                                                                              batch_size=batch_size,
                                                                              learning_rate=learning_rate,
                                                                              threshold=threshold,
                                                                              max_vocab_size=max_vocab_size,
                                                                              embed_size=embed_size,
                                                                              print_every_step=print_every_step,
                                                                              dropout=dropout,
                                                                              learning_rate_max_offset=learning_rate_max_offset,
                                                                              fine_tuning_epochs=fine_tuning_epochs,
                                                                              max_seq_len=max_seq_len)
            scores_va.append(
                [f1_score(targets_va, predictions_va > threshold), roc_auc_score(targets_va, predictions_va),
                 precision_score(targets_va, predictions_va > threshold),
                 recall_score(targets_va, predictions_va > threshold)])
            scores_te.append([f1_score(targets_te, predictions_te > best_threshold),
                              roc_auc_score(targets_te, predictions_te),
                              precision_score(targets_te, predictions_te > best_threshold),
                              recall_score(targets_te, predictions_te > best_threshold)])
    if len(scores_te) == 1:
        scores_te = scores_te[0]
    if len(scores_va) == 1:
        scores_va = scores_va[0]
    return {'scores_te': scores_te, 'scores_va': scores_va}
