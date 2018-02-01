"""
The default symbolic data for Q18 is a bit strange - all the ">" questions have
a "true" answer, all the "<" questions have a "false" answer.

This makes the task trivial. 

This script is created to fix this, by changing all the ">" questions to "<" 
questions and randomly flipping the pairs being compared, thus "true" and 
"false" answers have roughly equal probability.

This is an overlooked issue in our initial ICLR submission.

Yujia Li, 04/2016
"""

import argparse
import random
import re

def fix_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    switch_prob = 0.5
    random.seed(1023)   # make sure each time we run this we get the same result

    p_eval = re.compile(r'(\d+) eval (\w+) ([><]) (\w+)(\s+)(\w+)([^\n]+)')

    with open(file_path, 'w') as f:
        for line in lines:
            m = p_eval.search(line)
            if m is not None:
                line_num, A, op, B, space, ans, others = m.groups()
                if op == '>':   # change all ">" to "<"
                    B, A = A, B
                    op = '<'

                if random.random() < switch_prob:
                    B, A = A, B
                    if ans == 'true':
                        ans = 'false'
                    else:
                        ans = 'true'

                line = '%s eval %s %s %s%s%s%s\n' % (line_num, A, op, B, space, ans, others)

            f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the 18.txt file to be fixed.')
    opt = parser.parse_args()

    fix_file(opt.file_path)

