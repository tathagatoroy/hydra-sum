'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
Code originally forked from hydrasum salesforce repo. Comments and docstrings added by BARD. So please proceed with caution about the comments
'''
from nltk import word_tokenize
import os
import csv


def get_extractive_fragments(article, summary):
    """
    Extracts fragments from an article that match sequences of words in a summary.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        list: A list of lists, where each sublist represents a sequence of word indexes
            in the article that match a sequence in the summary.
        list: The tokenized article.
        list: The tokenized summary.
    """

    article_tokens = word_tokenize(article.lower())
    summary_tokens = word_tokenize(summary.lower())

    F = []  # List to store the extracted fragments
    i, j = 0, 0  # Indexes for iterating over article and summary tokens, respectively

    while i < len(summary_tokens):
        f = []  # List to store the current fragment
        while j < len(article_tokens):
            if summary_tokens[i] == article_tokens[j]:
                i_, j_ = i, j  # Store starting indexes of potential fragment
                while summary_tokens[i_] == article_tokens[j_] and (
                    i_ < len(summary_tokens) and j_ < len(article_tokens)
                ):
                    i_, j_ = i_ + 1, j_ + 1  # Update indexes while words match
                if len(f) < (i_ - i):  # Update fragment if a longer match is found
                    f = list(range(i, i_))
                j = j_  # Set j to the next position after the matched sequence
            else:
                j += 1  # Move to the next article token if no match found
        i += max(len(f), 1)  # Update i by the length of the extracted fragment or 1
        j = 1  # Reset j for the next iteration

        F.append(f)  # Append the extracted fragment to the list

    return F, article_tokens, summary_tokens


def get_extractive_coverage(article, summary):
    """
    Calculates the extractive coverage of a summary on an article.

    Coverage is defined as the ratio of words in the summary covered by fragments
    extracted from the article.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        float: The extractive coverage of the summary on the article.
    """

    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    coverage = float(sum([len(f) for f in frags])) / float(len(summary_tokens))
    return coverage


def get_fragment_density(article, summary):
    """
    Calculates the fragment density of a summary on an article.

    Density is defined as the average squared length of extracted fragments.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        float: The fragment density of the summary on the article.
    """

    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    density = float(sum([len(f)**2 for f in frags])) / float(len(summary_tokens))
    return density


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """
    Reads a tab separated value file.

    Args:
        input_file (str): The path to the TSV file.
        quoting (Optional[int]): The quoting style for the CSV reader.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
    """

    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


if __name__ == "__main__":
    input_file = "../../data/newsroom/mixed/train.tsv"
    data = _read_tsv(input_file)

    for d in data:
        article = d["article"]
        summary = d["summary"]

        density = get_fragment_density(article, summary)
        coverage = get_extractive_coverage(article, summary)


        break
