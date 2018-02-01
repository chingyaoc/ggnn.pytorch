"""
Preprocess the bAbI symbolic data into a nicer form.

Yujia Li, 10/2015
"""

import os
import sys
import argparse

def parse_dataset(file_name, edge_type_dict={}, node_id_dict={}, question_type_dict={}, label_dict={}):
    """
    Parse the dataset file.

    Return dataset, edge_type_dict, node_id_dict, question_type_dict, label_dict

    dataset: a list of (edges, questions) tuples, edges is a list of edges of
        form (src_id, edge_type, tgt_id). questions is a list of questions of
        form (question_type, arg1, ..., argn[, label]).  If the last label is
        given then the question is a graph-level prediction task, otherwise
        the task is node selection and the last argument is the target node.

    edge_type_dict, node_id_dict, question_type_dict, label_dict: 
        str -> ID dictionaries.
    """
    dataset = []
    with open(file_name, 'r') as f:
        edges = []
        questions = []
        prev_id = 0
        for line in f:
            tokens = line.split()
            line_id = int(tokens[0])
            if line_id < prev_id:
                dataset.append((edges, questions))
                edges = []
                questions = []

            if len(tokens) == 4:
                # edge line
                src = tokens[1]
                etype = tokens[2]
                tgt = tokens[3]

                if src not in node_id_dict:
                    src_id = len(node_id_dict) + 1
                    node_id_dict[src] = src_id
                else:
                    src_id = node_id_dict[src]

                if tgt not in node_id_dict:
                    tgt_id = len(node_id_dict) + 1
                    node_id_dict[tgt] = tgt_id
                else:
                    tgt_id = node_id_dict[tgt]

                if etype not in edge_type_dict:
                    edge_type = len(edge_type_dict) + 1
                    edge_type_dict[etype] = edge_type
                else:
                    edge_type = edge_type_dict[etype]

                edges.append((src_id, edge_type, tgt_id))
            else:
                # question line
                if tokens[2] == 'path':
                    # path question, task 19
                    qtype = tokens[2]
                    src = tokens[3]
                    tgt = tokens[4]
                    label_str = tokens[5]

                    labels = label_str.split(',')

                    for i in xrange(len(labels)):
                        if labels[i] not in label_dict:
                            label = len(label_dict) + 1
                            label_dict[labels[i]] = label
                        else:
                            label = label_dict[labels[i]]

                        labels[i] = label
                else:
                    # questions should have form <head_word> src qtype tgt [label]
                    src = tokens[2]
                    qtype = tokens[3]
                    tgt = tokens[4]
                    label_str = tokens[5] if not tokens[5].isdigit() else None
                    
                    if label_str is not None:
                        if label_str not in label_dict:
                            label = len(label_dict) + 1
                            label_dict[label_str] = label
                        else:
                            label = label_dict[label_str]
                    else:
                        label = None

                src_id = node_id_dict[src]
                tgt_id = node_id_dict[tgt]
                
                if qtype not in question_type_dict:
                    question_type = len(question_type_dict) + 1
                    question_type_dict[qtype] = question_type
                else:
                    question_type = question_type_dict[qtype]

                if tokens[2] == 'path':
                    questions.append([question_type, src_id, tgt_id] + labels)
                else:
                    questions.append((question_type, src_id, tgt_id, label) if label is not None else (question_type, src_id, tgt_id))

            prev_id = line_id

        if len(edges) > 0 and len(questions) > 0:
            dataset.append((edges, questions))

    return dataset, edge_type_dict, node_id_dict, question_type_dict, label_dict

def write_dataset(dataset, edge_type_dict, node_id_dict, question_type_dict, label_dict, output_dir, output_prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_dict(edge_type_dict,     os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'edge_types')))
    write_dict(node_id_dict,       os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'node_ids')))
    write_dict(question_type_dict, os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'question_types')))
    write_dict(label_dict,         os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'labels')))

    write_examples(dataset, os.path.join(output_dir, '%s_%s.txt' % (output_prefix, 'graphs')))

def write_examples(dataset, output_file):
    with open(output_file, 'w') as f:
        for e, q in dataset:
            for src, etype, tgt in e:
                f.write('%d %d %d\n' % (src, etype, tgt))
            for q_args in q:
                f.write('?')
                for arg in q_args:
                    f.write(' %d' % arg)
                f.write('\n')
            f.write('\n')

def write_dict(d, output_file):
    with open(output_file, 'w') as f:
        for k,v in d.iteritems():
            f.write('%s=%s\n' % (str(k), str(v)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question_id', type=int, help='ID of the question to process. We only use {4,15,16,18,19}')
    parser.add_argument('input_dir', default='symbolic', help='Path to the directory that contains generated raw symbolic data, should contain two directories train and test.')
    parser.add_argument('output_dir', default='processed', help='Path to the directory to store processed symbolic data.')

    opt = parser.parse_args()

    question_id = opt.question_id
    input_dir = opt.input_dir
    output_dir = opt.output_dir

    d, e, n, q, l = parse_dataset(os.path.join(input_dir, 'train', '%s.txt' % question_id))
    write_dataset(d, e, n, q, l, os.path.join(output_dir, 'train'), question_id)

    d, e, n, q, l = parse_dataset(os.path.join(input_dir, 'test', '%s.txt' % question_id), e, n, q, l)
    write_dataset(d, e, n, q, l, os.path.join(output_dir, 'test'), question_id)

