#!/usr/bin/python3

import sys

min = 10
max = -10

lines = sys.stdin.readlines()
for i in range(len(lines)):
    n = float(lines[i])
    if n > max:
        max = n
    if n < min:
        min = n

print('Max: {max}, Min: {min}'.format(max=max, min=min))
