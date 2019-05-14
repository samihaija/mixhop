

import pickle
import random

import matplotlib.pyplot as plt
import numpy

from absl import app
from absl import flags
from absl import logging
import networkx as nx

flags.DEFINE_string('graph', 'ind.n5000-h0.3-c10.graph', '')
flags.DEFINE_bool('plot', False, 'If set, pdf will be saved.')

FLAGS = flags.FLAGS


def main(_):
  extensionless = FLAGS.graph.replace('.graph', '')
  ally_file = FLAGS.graph.replace('.graph', '.ally')
  graph = pickle.load(open(FLAGS.graph))
  ally = numpy.load(ally_file)
  
  num_classes = ally.shape[1]
  variance_factor = 350
  start_cov = numpy.array(
      [[70.0 * variance_factor, 0.0],
       [0.0, 20.0 * variance_factor]])

  cov = start_cov
  theta = numpy.pi*2/num_classes
  rotation_mat = numpy.array(
      [[numpy.cos(theta), -numpy.sin(theta)],
       [numpy.sin(theta), numpy.cos(theta)]])
  radius = 300
  allx = numpy.zeros(shape=[len(ally), 2], dtype='float32')
  plt.figure(figsize=(40,40))
  for cls, theta in enumerate(numpy.arange(0, numpy.pi*2, numpy.pi*2/num_classes)):
    gaussian_y = radius * numpy.cos(theta)
    gaussian_x = radius * numpy.sin(theta)
    num_points = numpy.sum(ally.argmax(axis=1) == cls)
    coord_x, coord_y = numpy.random.multivariate_normal(
        [gaussian_x, gaussian_y], cov, num_points).T
    cov = rotation_mat.T.dot(cov.dot(rotation_mat))

    # Belonging to class cls
    example_indices = numpy.nonzero(ally[:, cls] == 1)[0]
    random.shuffle(example_indices)
    allx[example_indices, 0] = coord_x
    allx[example_indices, 1] = coord_y
    #plt.plot(coord_x, coord_y, 'x');

  numpy.save(
      open(extensionless + '.allx', 'w'),
      allx)

  if FLAGS.plot:
    g = nx.Graph(graph)
    edge_list = list(g.edges())
    num_edges = len(edge_list)
    permutation = numpy.random.permutation(num_edges)
    random.shuffle(permutation)

    for v1, v2 in [edge_list[i] for i in permutation[:400]]:
      xx = [allx[v1][0], allx[v2][0]]
      yy = [allx[v1][1], allx[v2][1]]
      plt.plot(xx, yy, 'b-')


    for cls, theta in enumerate(numpy.arange(0, numpy.pi*2, numpy.pi*2/num_classes)):
      example_indices = numpy.nonzero(ally[:, cls] == 1)[0]
      plt.plot(allx[example_indices, 0], allx[example_indices, 1], 'o')
    output_filename = extensionless + '.depiction.pdf'
    plt.savefig(extensionless + '.depiction.pdf')
    print('Wrote: ' + output_filename)

  return 0


if __name__ == '__main__':
  app.run(main)
