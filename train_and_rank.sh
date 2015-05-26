#!/bin/bash

java -jar RankLib.jar -train ./data/train_feature -ranker 0 -tree 50 -save ./model/model.txt
java -jar RankLib.jar -rank ./data/test_feature -load ./model/model.txt -score score.txt
