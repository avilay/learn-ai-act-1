#!/bin/zsh
python -m multi.main dist.rank=0 &
python -m multi.main dist.rank=1 &
python -m multi.main dist.rank=2 &
