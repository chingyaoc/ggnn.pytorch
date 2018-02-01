"""
Preprocess data into a nicer form.

Yujia Li, 11/2015
"""

import os
import sys

def parse_dataset(file_name, edge_type_dict={}, question_type_dict={}):
    dataset = []
    with open(file_name, 'r') as f:
        edges = []
        questions = []
        for line in f:
            tokens = line.split()
            if tokens[0] == 'eval':
                qtype = tokens[1]
                args = tokens[2:-1]
                targets = tokens[-1]

                if qtype not in question_type_dict:
                    question_type = len(question_type_dict) + 1
                    question_type_dict[qtype] = question_type
                else:
                    question_type = question_type_dict[qtype]

                args = [int(a)+1 for a in args]
                targets = [int(t)+1 for t in targets.split(',')]

                questions.append((question_type, args, targets))
                dataset.append((edges, questions))

                edges = []
                questions = []
            else:
                src = tokens[0]
                etype = tokens[1]
                tgt = tokens[2]

                if etype not in edge_type_dict:
                    edge_type = len(edge_type_dict) + 1
                    edge_type_dict[etype] = edge_type
                else:
                    edge_type = edge_type_dict[etype]

                edges.append((int(src)+1, edge_type, int(tgt)+1))

    return dataset, edge_type_dict, question_type_dict

def write_dict(d, output_file):
    with open(output_file, 'w') as f:
        for k,v in d.iteritems():
            f.write('%s=%s\n' % (str(k), str(v)))

def write_examples(dataset, output_file):
    with open(output_file, 'w') as f:
        for e, q in dataset:
            for src, etype, tgt in e:
                f.write('%d %d %d\n' % (src, etype, tgt))

            for q_args in q:
                f.write('?')
                f.write(' %d ' % q_args[0])  # qtype
                for a in q_args[1]:
                    f.write(' %d' % a)
                f.write(' ')
                for t in q_args[2]:
                    f.write(' %d'% t)
                f.write('\n')
            f.write('\n')

def write_dataset(dataset, edge_type_dict, question_type_dict, output_dir, output_prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_dict(edge_type_dict,     os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'edge_types')))
    write_dict(question_type_dict, os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'question_types')))

    write_examples(dataset, os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'graphs')))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python %s <question_id> [<input_dir>] <output_dir>' % (os.path.basename(__file__)))
    else:
        question_id = sys.argv[1]
        output_dir = sys.argv[2]
        input_dir = 'data'
        if len(sys.argv) >= 4:
            output_dir = sys.argv[3]
            input_dir = sys.argv[2]

        d, e, q = parse_dataset('%s/train/%s.txt' % (input_dir, question_id))
        write_dataset(d, e, q, '%s/train' % output_dir, question_id)

        d, e, q = parse_dataset('%s/test/%s.txt' % (input_dir, question_id), e, q)
        write_dataset(d, e, q, '%s/test' % output_dir, question_id)

