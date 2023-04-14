#!/usr/bin/env python3

import argparse
import logging
import json
import os
import concurrent.futures
from html.parser import HTMLParser
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

from tqdm import tqdm
import openai

logger = logging.getLogger("base")

# TODO: use yaml instead of json? Use slightly less tokens
REQUESTED_STRUCTURE = """\
{
"name": STR, // company or institution name
"locations": ARRAY[STR], // company location, countries or cities
"visa": BOOL, // True if the company handle visas. When only "VISA" is written sponsorship is implied
"remote work policy": STR, // one of `onsite`, `hybrid`, `remote`, `world-wide remote`. If multiple, only write the most flexible
"apply through": STR, // how to apply, should be an email address or a URL to a job board or company's website
"sector": STR, // sector or industry the company is focusing on
"technologies": ARRAY[STR], // key technologies being used at the company. Can be generic (example `ML`) or specific (example `Haskell`). Maximum 5
"roles": ARRAY[{ // roles the company is looking for
 "job_title": STR, // job title
 "seniority": ARRAY[STR], // expected seniority, array as sometimes multiple levels are mentionned
 "salary": INT, // salary. Expand if necessary, example 90K become 90000. If salary is a range, take the lower bound.
 "currency": STR, // salary currency, example USD, EUR, GBP, JPY
 "incentives": STR, // incentives on top of the base salary
 "employment type": STR // either `full-time`, `part-time` or `contractor`
}]
}
"""

BASE_PROMPT = f"""\
You will extract key informations as a compact json structured as follow:

// comment lines like this are explanations and will be omitted from your output
// `null` can be used for fields not specified in the posting (or [] if the expected type is ARRAY)
{REQUESTED_STRUCTURE}

First posting:
"""

SYSTEM_PROMPT = "You are JobAssistant, an AI processing online job postings."


class Parser(HTMLParser):
    """Basic parser, should suffice for what's accepted in HN:
    https://news.ycombinator.com/formatdoc"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._content = ""

    def handle_data(self, data):
        self._content += data

    def handle_starttag(self, tag, _):
        if tag == "p":
            self._content += "\n"

    @property
    def content(self):
        c, self._content = self._content.rstrip(), ""
        return c

    def __call__(self, data: str) -> str:
        self.feed(data)
        return self.content


@dataclass
class ProcessOutput:
    base_prompt: str
    expected_format: str
    output_file: Path
    # 0 is deterministic, 1 more random (default), max 2.0
    temperature: float = 0.5
    html_parser: Parser = Parser()

    def __post_init__(self):
        self._validate_against = json.loads(
            "\n".join([t.rsplit(" //", maxsplit=1)[0] for t in self.expected_format.split("\n")])
            .replace("ARRAY[STR]", "[]")
            .replace("ARRAY", "")
            .replace("STR", '""')
            .replace("INT", "42")
            .replace("BOOL", "true")
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
        with self.output_file.open("a") as f:
            f.write(json.dumps(data) + "\n")

    def chat_completion(self, text) -> Dict:
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=1024,
            # 0 is deterministic, 1 more random (default), max 2.0
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.base_prompt + "\n" + text},
            ],
        )

        r = r.to_dict()  # type:ignore
        r["usage"] = r["usage"].to_dict()
        r["choices"][0] = r["choices"][0].to_dict()
        r["choices"][0]["message"] = r["choices"][0]["message"].to_dict()

        return r

    def decode_model_output(self, model_output):
        r = {}
        try:
            r["model_output"] = json.loads(model_output)
        except json.JSONDecodeError:
            r["model_output"] = {}
            r["status"] = "JSONDecodeError"
        else:
            try:
                r["status"] = "OK" if self._validate_schema(r["model_output"]) else "invalid"
            except Exception as e:
                r["status"] = "meh"
                logger.warning(f"Invalid schema:\n{e}")

        return r

    def _process(self, source_comment) -> Dict:
        comment = self.html_parser(source_comment["text"])

        response = self.chat_completion(comment)

        row = (
            {
                "hn_id": source_comment["id"],
                "hn_text": source_comment["text"],
            }
            | response["usage"]
            | self.decode_model_output(response["choices"][0]["message"]["content"])
        )

        self.append_to_file(row)

        return row

    def __call__(self, source_comment):
        # TODO: handle failures elegantly
        try:
            return self._process(source_comment)
        except Exception as _:
            logger.warning(f"Failed on:\n{source_comment}")


def tqdm_parallel_map(executor, fn, *iterables):
    futures_list = []

    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]

    # TODO: show running cost in tqdm bar
    for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list)):
        yield f.result()


def main(comments_file: Path, output: Path, max_parallel_requests: int) -> None:
    logger.debug("Starting")

    with comments_file.open() as f:
        all_comments = json.load(f)

    # Ignore deleted comments
    all_comments = [x for x in all_comments if not "deleted" in x.keys()]

    # Load API key
    openai.api_key = os.getenv(
        "OPENAI_API_KEY",
        default=Path.read_text(Path.home() / ".ssh" / "openai_api").rstrip(),
    )

    # Process all comments
    processor = ProcessOutput(BASE_PROMPT, REQUESTED_STRUCTURE, output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        _ = list(tqdm_parallel_map(executor, processor, all_comments))

    logger.info("Processed all comments, checking results")

    results = [json.loads(x) for x in Path("output.jsonl").read_text().splitlines()]

    # Cost as of 2023/04/14 is USD 0.002 per 1K token
    cost_per_token = 0.002 / 1000
    logger.info(
        "Processed {} comments, cost (in USD): {}\n".format(
            len(results), sum(x["total_tokens"] for x in results) * cost_per_token
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use GPT-3.5 to process HN job listings",
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

    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter("{asctime} {levelname} {name:>16}│ {message}", datefmt="%H:%M:%S", style="{"))
    logger.addHandler(ch)

    # Pretty colors, essential
    _levels = [[226, "DEBUG"], [148, "INFO"], [208, "WARNING"], [197, "ERROR"]]
    for color, lvl in _levels:
        _l = getattr(logging, lvl)
        logging.addLevelName(_l, "\x1b[38;5;{}m{:<7}\x1b[0m".format(color, logging.getLevelName(_l)))

    main(**args)
