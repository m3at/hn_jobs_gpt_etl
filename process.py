#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
from collections.abc import Generator
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from textwrap import shorten
from typing import Final, Tuple

from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger("base")


# As of 2024/07/24, price per 1M tokens
# https://openai.com/api/pricing/
_PRICES = {
    # Input, output
    "gpt-4o": (5, 15),
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "gpt-4o-mini": (0.15, 0.6),
}
# Prices are per 1M tokens
PRICES: Final[dict[str, Tuple[int, int]]] = {
    k: (v[0] / 1e6, v[1] / 1e6) for k, v in _PRICES.items()
}


# MODEL: Final = "gpt-3.5-turbo-0125"
# MODEL: Final = "gpt-4o"
MODEL: Final = "gpt-4o-mini"
assert MODEL in PRICES


# cat output.jsonl | jq -c '.model_output.roles[].seniority[]' | sort | uniq
REQUESTED_STRUCTURE = """\
{
"company": STR,
"locations": ARRAY[STR],
"visa": BOOL, // True only when visa or immigration sponsorship is explicitely mentionned
"remote": LITERAL[onsite | hybrid | remote | world-wide remote], // If multiple, the most flexible is selected
"application": STR, // How to apply. Can be an email address or the URL for a job board or company's website
"sector": STR, // Sector or industry the company is focusing on
"technologies": ARRAY[STR], // Key technologies being used. Can be generic (example `ML`) or specific (example `Haskell`). Maximum 5
"roles": ARRAY[{ // Roles the company is hiring for. Deduplicate closely related roles.
 "job_title": STR,
 "seniority": LITERAL[junior | mid | senior | staff | executive], // Expected seniority, using the highest if multiples. Map to closest, or null if too broad like "all levels"
 "salary": INT, // Expand if necessary, ex. 90K become 90000. If range, use lower bound.
 "currency": STR, // ex. USD, EUR, GBP, JPY
 "incentives": STR, // Notable incentives. Ignore vague things like "competitive", and generic or minor ones like transportation, medical and office supply
 "employment": LITERAL[full-time | part-time | contractor]
}]
}
"""


SYSTEM_PROMPT = f"""\
You are JobAssistant, an AI processing online job postings, only outputing JSON.
You extract key informations from raw postings, structuring it as follow:

// comment lines like this are explanations and will be omitted from your output
// `null` is used for information missing in the posting (or [] if the expected type is ARRAY)
{REQUESTED_STRUCTURE}
"""


class Parser(HTMLParser):
    """Basic parser, should suffice for what's accepted in HN:
    https://news.ycombinator.com/formatdoc"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._content = ""
        self._current_link = None

    def handle_data(self, data):
        if self._current_link:
            self._content += self._current_link
            self._current_link = None
        else:
            self._content += data

    def handle_starttag(self, tag, attrs):
        # Paragraphs as simple line break
        if tag == "p":
            self._content += "\n"
        # Replace the potentially truncated link in text with full url in the tag
        elif tag == "a":
            for attr, value in attrs:
                if attr == "href":
                    if value is not None:
                        # up to a max size
                        self._current_link = shorten(
                            value, width=192, placeholder="..."
                        )
                    break

    def handle_endtag(self, tag):
        if tag == "a" and self._current_link:
            self._content += self._current_link
            self._current_link = None

    @property
    def content(self):
        c, self._content = self._content.rstrip(), ""
        return c

    def __call__(self, data: str) -> str:
        self.feed(data)
        return self.content


def get_cost(usage: CompletionUsage, *, prices=PRICES[MODEL]) -> float:
    a, b = prices
    return (a * usage.prompt_tokens) + (b * usage.completion_tokens)


def threads_progress(executor, fn, *iterables) -> Generator[float, None, None]:
    """Progress bar showing the running cost."""
    futures_list = []

    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]

    # tqdm.contrib.concurrent.thread_map is also nice, and a single line
    running_cost = 0
    with Progress(
        TextColumn("total: {task.description} USD"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(elapsed_when_finished=True),
    ) as progress:
        task = progress.add_task(" " * 5, total=len(futures_list))

        for f in concurrent.futures.as_completed(futures_list):
            c_single = f.result()
            running_cost += c_single
            progress.update(task, description=f"{running_cost:>5.2f}", advance=1)
            yield c_single


@dataclass
class ProcessOutput:
    expected_format: str
    output_file: Path
    temperature: float = 0.0  # 0 is deterministic, 1 more random (default), max 2.0
    html_parser: Parser = Parser()

    def __post_init__(self):
        self.file_lock = threading.Lock()

        self._validate_against = json.loads(
            "\n".join(
                [
                    t.rsplit(" //", maxsplit=1)[0]
                    for t in self.expected_format.split("\n")
                ]
            )
            .replace("ARRAY[STR]", "[]")
            .replace("ARRAY", "")
            .replace("STR", '""')
            .replace("LITERAL[onsite | hybrid | remote | world-wide remote]", '""')
            .replace("LITERAL[junior | mid | senior | staff | executive]", '""')
            .replace("LITERAL[full-time | part-time | contractor]", '""')
            .replace("INT", "42")
            .replace("BOOL", "true")
        )

        api_key = os.environ.get("OPENAI_API_KEY", None)
        # assert api_key is not None, "Please set the env OPENAI_API_KEY"
        self.client = OpenAI(
            # base_url="http://localhost:8080/v1",
            api_key=api_key,
        )

    def _validate_schema(self, completion) -> bool:
        """Basic validation of keys only"""

        keys_roles = set(self._validate_against["roles"][0].keys())
        return all(
            [
                set(self._validate_against.keys()) == set(completion.keys()),
            ]
            + [keys_roles == set(r.keys()) for r in completion["roles"]]
        )

    def append_to_file(self, data):
        with self.file_lock, self.output_file.open("a") as f:
            f.write(json.dumps(data) + "\n")

    @retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3))
    def call_api(self, user_message: str) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            timeout=30,
        )

    def chat_completion(self, text) -> Tuple[CompletionUsage, str]:
        completion = self.call_api(text)
        content = completion.choices[0].message.content
        usage = completion.usage

        assert usage is not None
        assert content is not None

        return usage, content

    def decode_model_output(self, model_output):
        r = {}
        try:
            r["model_output"] = json.loads(model_output)
        except json.JSONDecodeError:
            r["model_output"] = {}
            r["status"] = "JSONDecodeError"
        else:
            try:
                r["status"] = (
                    "OK" if self._validate_schema(r["model_output"]) else "invalid"
                )
            except Exception as e:
                r["status"] = "meh"
                logger.warning(f"Invalid schema:\n{e}")

        return r

    def _process(self, source_comment) -> float:
        # Transform markup to plain text
        comment = self.html_parser(source_comment["text"])

        # Get model output
        # response = self.chat_completion(comment)
        usage, response = self.chat_completion(comment)
        cost = get_cost(usage)

        # Merge source infos, response and status
        row = (
            {
                "hn_id": source_comment["id"],
                "hn_text": comment,
            }
            | usage.to_dict()
            | self.decode_model_output(response)
        )

        self.append_to_file(row)

        return cost

    def __call__(self, source_comment):
        # TODO: handle failures elegantly
        try:
            return self._process(source_comment)
        except Exception:
            logger.warning(f"Failed on:\n{source_comment}")


def main(comments_file: Path, output: Path, max_parallel_requests: int) -> None:
    all_comments = json.loads(comments_file.read_text())

    # Ignore deleted comments
    all_comments = [
        x for x in all_comments if all(k not in x.keys() for k in ["dead", "deleted"])
    ]

    processor = ProcessOutput(REQUESTED_STRUCTURE, output)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_parallel_requests
    ) as executor:
        costs = list(threads_progress(executor, processor, all_comments))

    logger.info(f"Total cost: {sum(costs):.2f} USD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use an LLM to process HN job listings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "comments_file",
        type=Path,
        help="json input file containing the comments to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="output.jsonl",
        help="Output file",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=128,
        help="Maximum requests to send in parallel",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    args = vars(parser.parse_args())
    log_level = getattr(logging, args.pop("log_level").upper())

    # Setup logging
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter(
            "{asctime} {levelname}│ {message}", datefmt="%H:%M:%S", style="{"
        )
    )
    logger.addHandler(ch)

    # Add colors if stdout is not piped
    if sys.stdout.isatty():
        _m = logging.getLevelNamesMapping()
        for c, lvl in [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]:
            logging.addLevelName(_m[lvl], f"\x1b[38;5;{c}m{lvl:<7}\x1b[0m")

    main(**args)
