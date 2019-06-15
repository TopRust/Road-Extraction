
import lmdb
import numpy as np

from chainer import dataset
class Dataset(dataset.DatasetMixin):
    

    def __init__(self, path_sat, path_map):
        
        self.env_sat = lmdb.open(path_sat) #env environment
        self.txn_sat = self.env_sat.begin(write=False, buffers=False) #txn 
        self.cur_sat = self.txn_sat.cursor()

        self.env_map = lmdb.open(path_map) #env environment
        self.txn_map = self.env_map.begin(write=False, buffers=False) #txn 
        self.cur_map = self.txn_map.cursor()

    def __len__(self):

        return self.env_sat.stat()['entries']

    def get_example(self, i):
        
        i = str(i).encode()
        image = np.fromstring(
            self.cur_sat.get(i), dtype=np.uint8).reshape((92, 92, 3))
        label = np.fromstring(
            self.cur_map.get(i), dtype=np.uint8).reshape((24, 24, 1))
        return image, label

def get_road():

    train = Dataset('data/mass_merged/lmdb/train_sat', 'data/mass_merged/lmdb/train_map')
    valid = Dataset('data/mass_merged/lmdb/valid_sat', 'data/mass_merged/lmdb/valid_map')

    return train, valid

