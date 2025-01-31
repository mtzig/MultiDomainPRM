from datasets import load_dataset
from tqdm import tqdm
import random
import json
import os
from bs4 import BeautifulSoup
import re
from collections import defaultdict

def strip_html(html_content):
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract and return plain text
    return soup.get_text()

def process_mc_qa(question, choices, answer_idx):
    '''
    question: the question string
    choices: list of the answer choices
    answer_idx: idx of the correct answer in choices

    returns question_processed, answer

    process multiple choice question into right format
    e.g.

    Question...?
    A. Choice 1
    B. Choice 2
    C. Choice 3

    ...

    '''
    letters = 'ABCDEFGHIJKLMNOPQ'

    choice_str = ''
    for idx, c in enumerate(choices):
        choice_str += f'\n{letters[idx]}. {c}'

    question_processed = question + choice_str

    answer = letters[answer_idx]

    return question_processed, answer

