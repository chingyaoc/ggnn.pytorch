"""
Preprocess data for RNN training.

Yujia Li, 10/2015
"""

import numpy as np
import argparse

def convert_graph_data(infile, outfile, n_val=0, n_train=0):
    data_list = []
    with open(infile, 'r') as f:
        edges = []
        questions = []
        for line in f:
            tokens = line.split()
            if len(tokens) == 0:
                data_list.append([edges, questions])
                edges = []
                questions = []
            else:
                if tokens[0] == '?':
                    questions.append(tokens[1:])
                else:
                    edges.append(tokens)

    if len(edges) > 0:
        data_list.append([edges, questions])

    if n_val == 0:
        if n_train == 0:
            write_data_list_to_file(data_list, outfile)
        else:
            np.random.shuffle(data_list)
            write_data_list_to_file(data_list[:n_train], outfile)
    else:
        np.random.shuffle(data_list)
        if n_train == 0:
            write_data_list_to_file(data_list[:-n_val], outfile)
        else:
            write_data_list_to_file(data_list[:n_train], outfile)
        write_data_list_to_file(data_list[-n_val:], outfile + '.val')

def write_data_list_to_file(data_list, filename):
    with open(filename, 'w') as f:
        for edges, questions in data_list:
            s_edges = ''
            for e in edges:
                s_edges += 'n' + e[0] + ' e' + e[1] + ' n' + e[2] + ' eol '

            for q in questions:
                s_q = 'q' + q[0]

                # allow at most 2 nodes
                for i in xrange(1, min(len(q) - 1, 3)):
                    s_q += ' n' + q[i]

                s_q += ' ans'
                # allow more than one answers, which will be interpreted as a sequence
                for i in xrange(min(len(q) - 1, 3), len(q)):
                    s_q += ' ' + q[i]

                f.write(s_edges + s_q + '\n')

def convert_rnn_data(infile, outfile, dictfile=None):
    """
    Convert each token in the example into an index to make processing easier.
    """
    d = {}
    if dictfile is not None:
        with open(dictfile, 'r') as f:
            for line in f:
                k, v = line.split()
                d[k] = int(v)

    next_idx = 1
    with open(outfile, 'w') as fout:
        with open(infile, 'r') as fin:
            for line in fin:
                tokens = line.split()

                in_targets = False
                for i in xrange(len(tokens)):
                    t = tokens[i]
                    if in_targets:
                        fout.write(' ' + t)
                        continue

                    if t in d:
                        idx = d[t]
                    else:
                        d[t] = next_idx
                        idx = next_idx
                        next_idx += 1

                    fout.write('%d ' % idx)

                    if t == 'ans':
                        in_targets = True
                        fout.write('')

                fout.write('\n')

                # fout.write(tokens[-1] + '\n')

    with open(outfile + '.dict', 'w') as f:
        for k, v in sorted(d.items(), key=lambda t: t[0]):
            f.write('%s %d\n' % (k, v))

if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='Convert graph data into standard form for RNNs.')
    cmd_parser.add_argument('infile', help='path to the input file that contains all the graphs')
    cmd_parser.add_argument('outfile', help='path to the output file to be created')
    cmd_parser.add_argument('--dict', help='path to an optional dictionary file', default=None)
    cmd_parser.add_argument('--mode', help='preprocessing mode', choices=['graph', 'rnn'], default='graph')
    cmd_parser.add_argument('--nval', help='number of examples to use for validation', type=int, default=0)
    cmd_parser.add_argument('--ntrain', help='number of examples to use for training', type=int, default=0)

    args = cmd_parser.parse_args()
    if args.mode == 'graph':
        convert_graph_data(args.infile, args.outfile, args.nval, args.ntrain)
    elif args.mode == 'rnn':
        convert_rnn_data(args.infile, args.outfile, args.dict)

