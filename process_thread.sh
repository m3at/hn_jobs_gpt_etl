#!/usr/bin/env bash

set -eo pipefail

PARENT_ITEM_ID="$1"
OUTPUT_NAME="$2.jsonl"

mkdir -p processed_comments/ archive/

./get_comments.sh "$PARENT_ITEM_ID"
# OpenAI quickly complains and drop responses at sensible rates, so keep it super low and just wait
./process.py "comments_$PARENT_ITEM_ID.json" --output processed_comments/"$OUTPUT_NAME" --max-parallel-requests 5
tar czf archive/"$OUTPUT_NAME".tar.gz processed_comments/"$OUTPUT_NAME"
