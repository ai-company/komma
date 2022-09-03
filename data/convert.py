import toml
import os
import sys
from os.path import join, dirname
import re
from alive_progress import alive_bar


PUNCT_MAP = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTION",
    '"': "QUOTE",
    "'": "SQUOTE",
    "-": "DASH",
    "!": "EXCLAMATION",
    ";": "SCOLON",
    ":": "COLON",
    "(": "LPAREN",
    ")": "RPAREN",
    "{": "LBRACE",
    "}": "RBRACE",
    "[": "LBRACKET",
    "]": "RBRACKET",
}

if __name__ == "__main__":
    config = toml.loads(open(sys.argv[1], "r", encoding="utf-8").read())

    data_path = join(dirname(__file__), "..", config["data"]["dataset_path"])

    f = open(data_path, "r")
    data = f.read()
    f.close()

    lines = data.split("\n")

    training = []
    validation = []

    with alive_bar(len(lines), title=f"Processing") as bar:
        for l, line in enumerate(lines):
            if len(line.strip()) == 0:
                continue  # skip empty lines

            result = []
            bar()

            words = line.split(" ")
            for w, word in enumerate(words):
                # next word is either capitalized or the last word, more or less correct, will fail in some cases
                is_last_word = (
                    w + 1 < len(words) and words[w + 1][0].isupper()
                ) or w == len(words) - 1
                # parsed = []

                tokens = list(
                    re.finditer(
                        r"(?P<NUMB>[0-9]+([,.][0-9]+)*)|(?P<WORD>\w+)|(?P<PUNC>\W)",
                        word,
                    )
                )

                # keep quoted words intact
                if len(tokens) > 1 and (
                    (tokens[0].group(0) == '"' and tokens[-1].group(0) == '"')
                    or (tokens[0].group(0) == "'" and tokens[-1].group(0) == "'")
                ):
                    result.append(("WORD", "".join(map(lambda x: x.group(0), tokens))))
                    # result.extend(parsed)
                    continue

                # parse words-with-dashes and number ranges as one
                has_words = False
                has_numbers = False
                has_dash = False
                poison = False

                for token in tokens:
                    lexeme_type = list(
                        {
                            k: v for k, v in token.groupdict().items() if v is not None
                        }.keys()
                    )[0]
                    token = token.group(0)

                    if token == "-":
                        has_dash = True
                    elif lexeme_type == "WORD":
                        has_words = True
                    elif lexeme_type == "NUMB":
                        has_numbers = True
                    else:
                        poison = True

                if has_dash and (has_words or has_numbers):
                    if not poison:
                        result.append(
                            ("WORD", "".join(map(lambda x: x.group(0), tokens)))
                        )
                        # result.extend(parsed)
                        # print(parsed)
                        continue
                    else:
                        groups = [""]

                        for token in tokens:
                            lexeme_type = list(
                                {
                                    k: v
                                    for k, v in token.groupdict().items()
                                    if v is not None
                                }.keys()
                            )[0]
                            token = token.group(0)

                            if (
                                token == "-"
                                or lexeme_type == "WORD"
                                or lexeme_type == "NUMB"
                            ):
                                groups[-1] += token
                            else:
                                groups.append((lexeme_type, token))
                                groups.append("")

                        for item in groups:
                            if type(item) != str:
                                if len(item) > 0:
                                    result.append(item)
                            else:
                                result.append(("WORD", item))

                        # result.extend(parsed)
                        # print(parsed)
                        continue

                # single quotes in words
                has_single_quote = False
                has_numbers = False
                has_words = False
                poison = False

                for token in tokens:
                    lexeme_type = list(
                        {
                            k: v for k, v in token.groupdict().items() if v is not None
                        }.keys()
                    )[0]
                    token = token.group(0)

                    if token == "'":
                        has_single_quote = True
                    elif lexeme_type == "WORD":
                        has_words = True
                    elif lexeme_type == "NUMB":
                        has_numbers = True
                    else:
                        poison = True

                if has_single_quote and (has_words or has_numbers):
                    if not poison:
                        result.append(
                            ("WORD", "".join(map(lambda x: x.group(0), tokens)))
                        )
                        # result.extend(parsed)
                        # print(parsed)
                        continue
                    else:
                        groups = [""]

                        for token in tokens:
                            lexeme_type = list(
                                {
                                    k: v
                                    for k, v in token.groupdict().items()
                                    if v is not None
                                }.keys()
                            )[0]
                            token = token.group(0)

                            if (
                                token == "'"
                                or lexeme_type == "WORD"
                                or lexeme_type == "NUMB"
                            ):
                                groups[-1] += token
                            else:
                                groups.append((lexeme_type, token))
                                groups.append("")

                        for item in groups:
                            if type(item) != str:
                                if len(item) > 0:
                                    result.append(item)
                            else:
                                result.append(("WORD", item))

                        # result.extend(parsed)
                        # print(parsed)
                        continue

                # parse words.with.dots and number ranges as one
                has_words = False
                has_numbers = False
                has_dot = False
                poison = False

                for token in tokens:
                    lexeme_type = list(
                        {
                            k: v for k, v in token.groupdict().items() if v is not None
                        }.keys()
                    )[0]
                    token = token.group(0)

                    if token == ".":
                        has_dot = True
                    elif lexeme_type == "WORD":
                        has_words = True
                    elif lexeme_type == "NUMB":
                        has_numbers = True
                    else:
                        poison = True

                if not is_last_word and has_dot and (has_words or has_numbers):
                    if not poison:
                        result.append(
                            ("WORD", "".join(map(lambda x: x.group(0), tokens)))
                        )
                        # result.extend(parsed)
                        # print(parsed)
                        continue
                    else:
                        groups = [""]

                        for token in tokens:
                            lexeme_type = list(
                                {
                                    k: v
                                    for k, v in token.groupdict().items()
                                    if v is not None
                                }.keys()
                            )[0]
                            token = token.group(0)

                            if (
                                token == "'"
                                or lexeme_type == "WORD"
                                or lexeme_type == "NUMB"
                            ):
                                groups[-1] += token
                            else:
                                groups.append((lexeme_type, token))
                                groups.append("")

                        for item in groups:
                            if type(item) != str:
                                if len(item) > 0:
                                    result.append(item)
                            else:
                                result.append(("WORD", item))

                        # result.extend(parsed)
                        # print(parsed)
                        continue

                # parse elements out as they are
                for w, token in enumerate(tokens):
                    lexeme_type = list(
                        {
                            k: v for k, v in token.groupdict().items() if v is not None
                        }.keys()
                    )[0]
                    token = token.group(0)

                    if lexeme_type == "PUNC":
                        if token in PUNCT_MAP:
                            result.append((PUNCT_MAP[token], token))
                        else:
                            result.append(("PUNC", token))
                    else:
                        result.append(("WORD", token))

                # print(parsed)
                # result.extend(parsed)

            # convert to model tokens, should handle more sentence endings
            condensed = []

            for (t, item) in result:
                if t == "WORD":
                    condensed.append(f"{item}\tO")
                elif t == "PERIOD":
                    if len(condensed) > 0:
                        parts = condensed[-1].split("\t")
                        condensed[-1] = f"{parts[0]}\tPERIOD"
                elif t == "COMMA":
                    if len(condensed) > 0:
                        parts = condensed[-1].split("\t")
                        condensed[-1] = f"{parts[0]}\tCOMMA"
                else:
                    condensed.append(f"{item}\tO")

            # we likely didn't see any valid sentence here, skip
            if len(condensed) == 0:
                continue

            # workaround missing termination for sentence endings, will have an issue for direct speech ."
            if condensed[-1].split("\t")[1] not in ["PERIOD"]:
                parts = condensed[-1].split("\t")
                condensed[-1] = f"{parts[0]}\tPERIOD"

            # print("\n".join(condensed))
            # break

            if l % 5 == 0:  # take every fifth line for validation
                validation.append("\n".join(condensed))
            else:
                training.append("\n".join(condensed))

    print("Writing to disk")
    with open(dirname(__file__) + "/train.txt", "w") as f:
        f.write("\n".join(training))
    with open(dirname(__file__) + "/val.txt", "w") as f:
        f.write("\n".join(validation))
