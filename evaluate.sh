#!/bin/bash

paste ./data/test_feature score.txt | awk 'BEGIN{sum1=0.0; sum2=0.0} function abs(x){return ((x < 0.0) ? -x : x)} {sum1 +=(abs($1-$NF) * abs($1-$NF)); sum2 += abs($1-$NF)} END{print "MSE: "sum1/7549.0"\nMAD: "sum2/7549.0}'
