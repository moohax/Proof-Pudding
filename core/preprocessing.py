# Text processing 
import re
from numpy import array
from collections import namedtuple
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse

ScoredObject = namedtuple('Scored', 'text, score')

# Returns two matricies (tokenized_text, scores) and the tokenizer for list of emails
# mode is passed to texts_to_matrix
def tokenize_data(scored_objects, mode, tokenizer = None):

    all_texts = [e.text for e in scored_objects]

    if not tokenizer:
        tokenizer = Tokenizer()

        print('[+] Fitting vocab')
        tokenizer.fit_on_texts(all_texts)

    if mode is 'lstm':
        text_sequences = tokenizer.texts_to_sequences(all_texts)
        text_matrix = pad_sequences(text_sequences, padding='pre')

    else:
        text_matrix = tokenizer.texts_to_matrix(all_texts, mode)
    
    score_matrix = array([e.score for e in scored_objects])

    return text_matrix, score_matrix, tokenizer

# Returns two matricies (tokenized_text, scores) and the tokenizer for list of links
# mode is passed to texts_to_matrix
def tokenize_links(scored_objects, mode, tokenizer = None):

    all_texts = [e.text for e in scored_objects]

    parsed_links = []
    for link in all_texts:
        url = urlparse(link)    
        domain = url.netloc

        if ':' in domain:
            #print('[!] domain has a port designation, skipping...')
            continue

        domain_parts = [p for p in re.split(r'\.|-', domain) if p]
        path_parts = [p for p in re.split(r'/|\\|-|_|&|#|=|\.', url.path) if p]
        query_parts = [p for p in re.split(r'&|=|\.', url.query) if p]

        parsed_links.append(domain_parts + path_parts + query_parts)

    if not tokenizer:
        tokenizer = Tokenizer()

        print('[+] Fitting vocab')
        tokenizer.fit_on_texts(parsed_links)

    if mode is 'lstm':
        text_matrix = tokenizer.texts_to_sequences(all_texts)
        text_matrix = pad_sequences(text_sequences, padding='pre')

    else:
        text_matrix = tokenizer.texts_to_matrix(all_texts, mode)
    
    score_matrix = array([e.score for e in scored_objects])

    return text_matrix, score_matrix, tokenizer