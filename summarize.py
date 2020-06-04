#!/usr/bin/env python
import sys
import json
from collections import defaultdict, OrderedDict
import numpy as np
import os
import socket
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('eval_file')
parser.add_argument('--order', type=str, choices=['min','max'], default='max')
parser.add_argument('--sort', action='store_true', default=True, help='sort')
args = parser.parse_args()



valid_string = 'valid'
test_string = 'test'

stats = defaultdict(list)

with open(args.eval_file, 'r') as fp:
    for line in fp:
        data = json.loads(line)
        for k, v in data.items():
            stats[k].append(v)

# Print summary
print('\nSummaries for {}'.format(sys.argv[1]))


def space(txt):
    fields = txt.split()
    fields = [fields[0].ljust(50)] + [f.ljust(7) for f in fields[1:]]
    return (' '.join(fields))


lines = []
print(space('Metric min min_epoch MAX MAX_epoch mean'))
for k, v in stats.items():
    # verify not interval
    if k.endswith('_interval'):
        continue

    min_epoch = np.argmin(v)
    max_epoch = np.argmax(v)
    
    # does it have interval ?
    v_interval = stats.get(k + '_interval', [0.] * len(v))

    txt = space('{} {:.4f}+{:.4f} (ep{}) {:.4f}+{:.4f} (ep{}) Mean {:.4f}'.format(k, v[min_epoch], v_interval[min_epoch], min_epoch, v[max_epoch], v_interval[max_epoch], max_epoch, np.mean(v)))
    if k.startswith(valid_string):
        k_test = test_string + k[len(valid_string):]
        if k_test in stats:
            v_test = stats[k_test]
            v_test_interval = stats.get(k_test + '_interval', [0.]*len(v_test))
            assert len(v_test) == len(v), '{} vs. {}'.format(len(v_test), len(v))
            assert len(v_test) == len(v_test_interval)
            txt += space('{} {:.4f}+{:.4f} (ep{}) {:.4f}+{:.4f} (ep{}) Mean {:.4f}'.format(k_test, v_test[min_epoch], v_test_interval[min_epoch], min_epoch, v_test[max_epoch], v_test_interval[max_epoch], max_epoch, np.mean(v_test)))
        else:
            txt += ' {} not_found'.format(k_test)

        lines.append((v[min_epoch], v[max_epoch], txt))

if args.sort:
    if args.order =='max':
        lines = sorted(lines, key=lambda a:a[1], reverse=True)
    else:
        lines = sorted(lines, key=lambda a:a[0], reverse=False)

for __, __, l in lines:
    print(l)
