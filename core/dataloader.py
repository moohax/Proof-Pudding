import base64
import csv

from alive_progress import alive_bar
from collections import namedtuple

ScoredObject = namedtuple('Scored', 'text, score')

def load_emails(csv_path):
    scored_emails = []

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_rows = [r for r in csv_reader][1:] # skip header

    print(f'[+] Emails to parse: {len(csv_rows)}')

    with alive_bar(len(csv_rows), bar='blocks') as bar:
        for row in csv_rows:
            bar()
            subject = base64.b64decode(row[0]).decode()
            body = base64.b64decode(row[1]).decode()
            score = int(row[12]) / 1000 # mlxlogscore

            final_text = subject + " " + body

            scored_emails.append(ScoredObject(final_text, score))

    print('')
    return scored_emails

def load_links(csv_path):

    csv_rows = []

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_rows = [r for r in csv_reader][1:] # skip header

    print(f'[+] Links to parse: {len(csv_rows)}')

    scored_links = []

    with alive_bar(len(csv_rows), bar='blocks') as bar:

        for row in csv_rows:
            bar()
            link = row[2]
            score = int(row[15])/1000 # mlxlogscore
            scored_links.append(ScoredObject(link, score))

    print('')
    return scored_links