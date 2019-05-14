import collections
import os
import pickle
import tensorflow as tf

import numpy
import scipy.sparse

def concatenate_csr_matrices_by_rows(matrix1, matrix2):
  """Concatenates sparse csr matrices matrix1 above matrix2.
  
  Adapted from:
  https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
  """
  new_data = numpy.concatenate((matrix1.data, matrix2.data))
  new_indices = numpy.concatenate((matrix1.indices, matrix2.indices))
  new_ind_ptr = matrix2.indptr + len(matrix1.data)
  new_ind_ptr = new_ind_ptr[1:]
  new_ind_ptr = numpy.concatenate((matrix1.indptr, new_ind_ptr))

  return scipy.sparse.csr_matrix((new_data, new_indices, new_ind_ptr))


def ReadDataset(dataset_dir, dataset_name):
  """Returns dataset files given e.g. ind.pubmed as a dataset_name.
 
  Args:
    dataset_dir: `data` directory of planetoid datasets.
    dataset_name: One of "ind.citeseer", "ind.cora", or "ind.pubmed".

  Returns:
    Dataset object (defined below).
  """
  base_path = os.path.join(dataset_dir, dataset_name)
  edge_lists = pickle.load(open(base_path + '.graph'))

  allx = numpy.load(base_path + '.allx') #.todense()
  ally = numpy.array(numpy.load(base_path + '.ally'), dtype='float32')

  # TODO(haija): Support Homophily Datasets [and upload them]
  if False: #FLAGS.homophily_dataset:
    # TODO(haija): Support Homophily and analysis of delta-op.
    if False:  #FLAGS.analyze_delta_opness:
      x_distances = []
      radius = 300
      angles = numpy.arange(0, numpy.pi*2, numpy.pi*2/50.0)
      for theta in angles:
        gaussian_y = radius * numpy.cos(theta)
        gaussian_x = radius * numpy.sin(theta)
        center = numpy.array([gaussian_x, gaussian_y])
        x_distances.append(
            numpy.expand_dims( ((allx - center)**2).sum(axis=1), 1))
      x_sims = numpy.concatenate(x_distances, axis=1)
      x_sims = numpy.exp(-1e-5*x_sims) 
      allx = x_sims

    llallx = scipy.sparse.csr_matrix(allx).tolil()

    num_train = len(ally) / 3
    # TODO Load from offline.
    test_idx = range(num_train*2, num_train*3)

  else:
    testx = numpy.load(base_path + '.tx') #.todense()

    # Add test
    test_idx = map(int, open(base_path + '.test.index').read().split('\n')[:-1])

    num_test_examples = max(test_idx) - min(test_idx) + 1
    sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                           dtype='float32')

    allx = concatenate_csr_matrices_by_rows(allx, sparse_zeros)
    llallx = allx.tolil()
    llallx[test_idx] = testx
    #allx = scipy.vstack([allx, sparse_zeros])

    test_idx_set = set(test_idx)


    testy = numpy.array(numpy.load(base_path + '.ty'), dtype='float32')
    ally = numpy.concatenate(
        [ally, numpy.zeros((num_test_examples, ally.shape[1]), dtype='float32')],
        0)
    ally[test_idx] = testy

  num_nodes = len(edge_lists)

  # Will be used to construct (sparse) adjacency matrix.
  edge_sets = collections.defaultdict(set)
  for node, neighbors in edge_lists.iteritems():
    edge_sets[node].add(node)   # Add self-connections
    for n in neighbors:
      edge_sets[node].add(n)
      edge_sets[n].add(node)  # Assume undirected.

  # Now, build adjacency list.
  adj_indices = []
  adj_values = []
  for node, neighbors in edge_sets.iteritems():
    for n in neighbors:
      adj_indices.append((node, n))
      adj_values.append(1 / (numpy.sqrt(len(neighbors) * len(edge_sets[n]))))

  adj_indices = numpy.array(adj_indices, dtype='int32')
  adj_values = numpy.array(adj_values, dtype='float32')
  return Dataset(
      num_nodes=num_nodes, edge_sets=edge_sets, test_indices=test_idx,
      adj_indices=adj_indices, adj_values=adj_values, allx=llallx, ally=ally)


class Dataset(object):
  """Dataset object giving access to sparse feature & adjacency matrices.

  Access the matrices, as a sparse tensor, using functions:
    Features: sparse_allx_tensor()
    Adjacency: sparse_adj_tensor().

  If you use these tensors in a tensorflow graph, you must supply their
  dependencies. In particular, feed them into your feed dictionary like:

  feed_dict = {}   # and populate it with your own placeholders etc.
  dataset.populate_feed_dict(feed_dict)  # makes adj and allx tensors runnable.
  """
  def __init__(self, allx=None, ally=None, num_nodes=None, test_indices=None,
               edge_sets=None, adj_indices=None, adj_values=None):
    self.allx = allx
    self.ally = ally
    self.num_nodes = num_nodes
    self.edge_sets = edge_sets
    self.adj_indices = adj_indices
    self.adj_values = adj_values
    self.sp_adj_tensor = None
    self.sp_allx_tensor = None
    self.test_indices = test_indices

  def populate_feed_dict(self, feed_dict):
    """Adds the adjacency matrix and allx to placeholders."""
    sp_adj_tensor = self.sparse_adj_tensor()
    feed_dict[self.x_indices_ph] = self.x_indices
    feed_dict[self.indices_ph] = self.adj_indices
    feed_dict[self.values_ph] = self.adj_values

  def sparse_allx_tensor(self):
    if self.sp_allx_tensor is None:
      xrows, xcols = self.allx.nonzero()
      self.x_indices = numpy.concatenate(
          [numpy.expand_dims(xrows, 1), numpy.expand_dims(xcols, 1)], axis=1)
      # TODO(haija): Support Homophily Datasets [and upload them]
      if False: # FLAGS.homophily_dataset:
        x_values = numpy.array(self.allx[xrows, xcols].todense(), dtype='float32')[0]

      else:
        x_values = tf.ones([len(xrows)], dtype=tf.float32)
      self.x_indices_ph = tf.placeholder(
          tf.int64, [len(xrows), 2], name='x_indices')
      dense_shape = self.allx.shape
      self.sp_allx_tensor = tf.SparseTensor(
          self.x_indices_ph, x_values, dense_shape)

    return self.sp_allx_tensor

  def sparse_adj_tensor(self):
    if self.sp_adj_tensor is None:
      self.indices_ph = tf.placeholder(
          tf.int64, [len(self.adj_indices), 2], name='indices')
      self.values_ph = tf.placeholder(
           tf.float32, [len(self.adj_indices)], name='values')
      dense_shape = [self.num_nodes, self.num_nodes]
      self.sp_adj_tensor = tf.SparseTensor(
          self.indices_ph, self.values_ph, dense_shape)

    return self.sp_adj_tensor

  def get_partition_indices(self, num_train_nodes, num_validate_nodes):
    train_indices = list(range(num_train_nodes))
    validate_indices = list(range(
        num_train_nodes, num_train_nodes + num_validate_nodes))
    test_indices = self.test_indices
    return train_indices, validate_indices, test_indices
