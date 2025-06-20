#!/usr/bin/env bash

python refactor_for_files.py \
    --repos 'llvm/llvm-project', 'openssl/openssl', 'redis/redis', 'bulletphysics/bullet3' \
    --path_tasks '../collection/data/tasks' \
    --path_refactored_tasks '../collection/data/refactored'