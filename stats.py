mongo_url = r'mongodb://user:password@mongoserver.com:port'
mongo_db = 'expr_insincere'
import numpy as np
from pymongo import MongoClient, DESCENDING
from scipy.stats import ttest_ind


def get_runs_of_id(db, id=85):
    result = list(db.runs.find({'_id': id}))[-1]
    return result


def extrac_scores(run_item):
    scores = run_item['result']['scores_va']
    f1_score = []
    auc_score = []
    for score in scores:
        f1_score.append(score[0])
        auc_score.append(score[1])
    return np.array(f1_score), np.array(auc_score)


def find_max_id(db):
    return list(db.runs.find().sort([('_id', DESCENDING), ]).limit(1))[-1]['_id']


def stats(diff_scores, n_splits=10, method='t test'):
    if method == 't test':
        centered_diff = np.array(diff_scores) - np.mean(diff_scores)
        t = np.mean(diff_scores) * (n_splits ** .5) / (np.sqrt(np.sum(centered_diff ** 2)+1e-7 / (n_splits - 1)))
        return t
    else:
        assert method == '5-by-2 paired t test'
        p_i1 = diff_scores[0]
        diff_scores = np.reshape(diff_scores, [5, 2])
        p_i = np.mean(diff_scores, axis=1, keepdims=True)
        s_i_square = np.sum((diff_scores - p_i) ** 2)
        return p_i1 / np.sqrt(np.mean(s_i_square))


def stats_test(db, id1=None, id2=780, method='t test'):
    # 731 larger hidden
    # 466 mixup +larger finetuning vocab + larger hidden + more routing in capsule + layer normalization + dropout
    # 361 mean glove  fasttext scaled
    # 314 with embedding mixup
    # 178 with fge
    # 130 with residual
    # 110 with dropout
    # 108 with embedding fine tuning
    # 102 baseline
    """
    compare results of two runs
    """
    if id1 is None:
        id1 = find_max_id(db)
    runs_1 = get_runs_of_id(db, id1)
    runs_2 = get_runs_of_id(db, id2)
    f1_scores1, auc_scores1 = extrac_scores(runs_1)
    f1_scores2, auc_scores2 = extrac_scores(runs_2)
    if len(f1_scores1) != len(f1_scores2):
        raise Exception('examples len not equal {} {}'.format(len(f1_scores1), len(f1_scores2)))
    print('f1 t-test:')
    f1_stats = stats(f1_scores1 - f1_scores2, method=method)
    print(f1_stats)
    print('auc t-test:')
    print(stats(auc_scores1 - auc_scores2, method=method))
    return f1_stats


if __name__ == '__main__':
    client = MongoClient(mongo_url)
    db = client[mongo_db]
    # print(find_max_id(db))
    f1_stats = stats_test(db)
