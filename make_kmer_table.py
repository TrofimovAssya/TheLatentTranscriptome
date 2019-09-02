import numpy as np
import pandas as pd
import os
import sys

fnames = os.listdir(sys.argv[1])
total = np.load(f'{sys.argv[1]}/{fnames[0]}')
for i in fnames[1:]:
    print (i)
    b = np.load(f'{sys.argv[1]}/{i}')
    total = np.hstack((total,b))

np.save('kmer_embeddings.npy',total)
