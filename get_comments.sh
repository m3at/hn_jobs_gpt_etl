#!/usr/bin/env bash

set -eo pipefail

PARENT_ITEM_ID="${1:-35424807}"
echo "Fetching for item=$PARENT_ITEM_ID"

# Get all root comments in the thread, then for each fetch the details
curl -s https://hacker-news.firebaseio.com/v0/item/$PARENT_ITEM_ID.json \
    | jq -r '.kids | .[]' \
    | xargs -I _ curl -s https://hacker-news.firebaseio.com/v0/item/_.json \
    | jq -s '.' > comments_$PARENT_ITEM_ID.json

