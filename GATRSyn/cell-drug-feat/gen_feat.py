import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from const import DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', nargs='+', type=str,
                    default=[r'OnePPI_sim_0.2_ge_(64x2)x384_2409052038',
                             r'OnePPI_sim_0.2_mut_(64x2)x384_2409052228'],
                    help="List of dirs that contains embeddings.npy ")
args = parser.parse_args()
print(args)
paths = dict()
embeddings = []
for d in args.dirs:
    f = os.path.join(DATA_DIR, d, 'embeddings384.npy')
    emb = np.load(f)
    embeddings.append(emb)
embedding = np.concatenate(embeddings, axis=1)
scaler = StandardScaler().fit(embedding)
embedding = scaler.transform(embedding)
print(embedding)
print(embedding.shape)
np.save(r'cell_feat.npy', embedding)


