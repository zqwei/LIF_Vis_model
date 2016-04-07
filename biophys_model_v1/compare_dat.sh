#!/bin/bash

# get comparing file names $1 $2

sort -n -k 1 -k 2 $1 > tmp1
sort -n -k 1 -k 2 $2 > tmp2
diff tmp1 tmp2
rm tmp1 tmp2
