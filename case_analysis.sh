#!/bin/bash

paste ./data/test_qid ./data/test_feature score.txt | awk -F '\t' 'function abs(x){return ((x<0.0)? -x:x)} {print abs($NF-$4), $2, $3, $NF, substr($4, 0, 1)}' | sort -gr | sed -n '1, 30p'
