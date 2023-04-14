# Extract structured data from HN job threads 

Extract structured data from free text in Hacker News [who is hiring monthly threads](https://news.ycombinator.com/item?id=35424807), using HN and OpenAI APIs.

![example of output after processing with pandas](./example_output.png)

_Example of output after processing with pandas_

Usage:
```bash
# HN item id of latest hiring thread
export THREAD_ID="35424807"

# Fetch comments from HN api
./get_comments.sh "$THREAD_ID"

# Set your OpenAI API key
export OPENAI_API_KEY="..."

# Process each comments, will save to output.jsonl
# Cost about $0.50 for 350 comments (size of latest HN thread)
process.py "comments_$THREAD_ID.json" [--output output.jsonl] [--max-parallel-requests 64]
```

Dependencies: `jq`, `openai`, `tqdm`
```bash
brew install jq
python3 -m pip install -U openai tqdm
```
