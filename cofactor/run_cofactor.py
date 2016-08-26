import itertools
import glob
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from scipy import sparse
import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')


sys.path.append('../../cofactor/src/')
import cofacto
import rec_eval 

DATA_DIR = '../data'

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

# read train data here
print('Read in data ...')
train_data = pd.read_csv('../data/aggregated_obs_2005.csv')
vad_data = pd.read_csv('../data/aggregated_obs_2006.csv')
test_data = pd.read_csv('../data/aggregated_obs_2007.csv')

print('Find the common stixel set ...')
train_index = train_data['INDEX']
vad_index = vad_data['INDEX']
test_index = test_data['INDEX']

common_index = set(train_index).intersection(set(vad_index)).intersection(set(test_index))
common_index = list(common_index)

print('Get observation matrices...')
def select_obs(data, stixel_set):
    selector = [list(data['INDEX']).index(x) for x in common_index]
    obs = data.iloc[selector, 1:]
    return sparse.csr_matrix(obs[obs > 0].as_matrix().astype(np.int8))
    
train_data = select_obs(train_data, common_index)
vad_data = select_obs(vad_data, common_index)
test_data = select_obs(test_data, common_index)

save_sparse_csr('../data/obs_train', train_data)
save_sparse_csr('../data/obs_vad', vad_data)
save_sparse_csr('../data/obs_test', test_data)

#train_data = load_sparse_csr('../data/obs_train.npz')
#vad_data = load_sparse_csr('../data/obs_vad.npz')
#test_data = load_sparse_csr('../data/obs_test.npz')

n_users, n_items = train_data.shape


watches_per_movie = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
print("The mean (median) watches per movie is %d (%d)" % (watches_per_movie.mean(), np.median(watches_per_movie)))

user_activity = np.asarray(train_data.sum(axis=1)).ravel()
print("The mean (median) movies each user wathced is %d (%d)" % (user_activity.mean(), np.median(user_activity)))


plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
plt.ylabel('Number of items that this user clicked on')
plt.xlabel('User rank by number of consumed items')
pass

plt.semilogx(1 + np.arange(n_items), -np.sort(-watches_per_movie), 'o')
plt.ylabel('Number of users who watched this movie')
plt.xlabel('Movie rank by number of watches')
pass

def _coord_batch(lo, hi, train_data):
    rows = []
    cols = []
    for u in xrange(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
    np.save(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
    pass


from joblib import Parallel, delayed

batch_size = 5000

start_idx = range(0, n_users, batch_size)
end_idx = start_idx[1:] + [n_users]

Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data) for lo, hi in zip(start_idx, end_idx))
pass


X = sparse.csr_matrix((n_items, n_items), dtype='float32')

for lo, hi in zip(start_idx, end_idx):
    coords = np.load(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)))
    
    rows = coords[:, 0]
    cols = coords[:, 1]
    
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_items, n_items), dtype='float32').tocsr()
    X = X + tmp
    
    print("User %d to %d finished" % (lo, hi))
    sys.stdout.flush()

np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'), X.data)
np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'), X.indices)
np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'), X.indptr)


float(X.nnz) / np.prod(X.shape)

## or co-occurrence matrix from the entire user history
#dir_predix = DATA_DIR
#
#
#data = np.load(os.path.join(dir_predix, 'coordinate_co_binary_data.npy'))
#indices = np.load(os.path.join(dir_predix, 'coordinate_co_binary_indices.npy'))
#indptr = np.load(os.path.join(dir_predix, 'coordinate_co_binary_indptr.npy'))
#
#X = sparse.csr_matrix((data, indices, indptr), shape=(n_items, n_items))
#
#float(X.nnz) / np.prod(X.shape)

def get_row(Y, i):
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]

count = np.asarray(X.sum(axis=1)).ravel()

n_pairs = X.data.sum()

M = X.copy()

for i in xrange(n_items):
    lo, hi, d, idx = get_row(M, i)
    M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))


M.data[M.data < 0] = 0
M.eliminate_zeros()

print float(M.nnz) / np.prod(M.shape)

# number of negative samples
k_ns = 1

M_ns = M.copy()

if k_ns > 1:
    offset = np.log(k_ns)
else:
    offset = 0.
    
M_ns.data -= offset
M_ns.data[M_ns.data < 0] = 0
M_ns.eliminate_zeros()


plt.hist(M_ns.data, bins=50)
plt.yscale('log')
pass



float(M_ns.nnz) / np.prod(M_ns.shape)


scale = 0.03
n_components = 20
max_iter = 20
n_jobs = 8
lam_theta = lam_beta = 1e-5 * scale
lam_gamma = 1e-5
c0 = 1. * scale
c1 = 10. * scale

save_dir = os.path.join(DATA_DIR, 'ML20M_ns%d_scale%1.2E' % (k_ns, scale))

reload(cofacto)
coder = cofacto.CoFacto(n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs, 
                        random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True, 
                        lam_theta=lam_theta, lam_beta=lam_beta, lam_gamma=lam_gamma, c0=c0, c1=c1)

coder.fit(train_data, M_ns, vad_data=vad_data, batch_users=5000, k=n_components)

test_data.data = np.ones_like(test_data.data)

n_params = len(glob.glob(os.path.join(save_dir, 'CoFacto_K' + str(n_components) + '_iter*.npz')))
#n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))

params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
U, V = params['U'], params['V']

print 'Test Recall@20: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=20, vad_data=vad_data)
print 'Test Recall@50: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=50, vad_data=vad_data)
print 'Test NDCG@100: %.4f' % rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)
print 'Test MAP@100: %.4f' % rec_eval.map_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)




