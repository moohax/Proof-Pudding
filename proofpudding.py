import sys

if sys.version_info.major != 3:
    print("[!] Python 3 is required")
    sys.exit(1)

import argparse
import csv
import pickle
import os
import numpy as np

from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from core import dataloader
from core import preprocessing
#from core import postprocessing
from core import generator
from core import models



Banner = """
            __.......__
        ,-"``           ``"-.
        |;------.-'      _.-'\\
        ||______|`  ' ' `    |
        ||------|            |
       _;|______|            |_
     (```"""""""|            |``)
     \'._       '-.........-'_.'/
      '._`""===........===""`_.'

            Proof Pudding
"""

class ProofPudding(object):

    def __init__(self, argv):
        print(Banner)

        parser = argparse.ArgumentParser(description="Playing with ProofPoint's ML models",)
        parser.add_argument('command', choices=['insights', 'score', 'train'], help='Operation to perform')

        if not argv:
            parser.print_help()
            return

        args = parser.parse_args(argv[0:1])
        getattr(self, args.command)(argv[1:])

    def insights(self, args):
        parser = argparse.ArgumentParser(
            prog=f'{sys.argv[0]} insights',
            description='Collect insights from a trained model')
        parser.add_argument('-m', '--model', required=True, help='Pickled model file to use (./models/*)')
        parser.add_argument('-d', '--data', required=True, choices=['links', 'texts'], help='Scored data type to use')
        parser.add_argument('output', help='Output path for CSV')
        args = parser.parse_args(args)
        
        model = load_model(args.model)

        with open(f'{args.model}.vocab', 'rb') as h:
            tokenizer = pickle.load(h)

        if args.data == 'texts':
            scored_emails = dataloader.load_emails('core/data/texts.csv')
            text_matrix, score_matrix, _ = preprocessing.tokenize_data(scored_emails, 'binary', tokenizer)
            
        elif args.data == 'links':
            scored_links = dataloader.load_links('core/data/links.csv')
            text_matrix, score_matrix, _ = preprocessing.tokenize_links(scored_links, 'binary', tokenizer)

        print(f'[+] Gathering insights ... (this could take a minute)')

        sorted_insights = models.get_insights(model, text_matrix, tokenizer)

        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')

            for token in sorted_insights:
                writer.writerow(token)

        print(f'[+] Wrote insights to {args.output}')
 

    def score(self, args):
        parser = argparse.ArgumentParser(
            prog=f'{sys.argv[0]} score',
            description='Score a text using a trained model')
        parser.add_argument('-m', '--model', required=True, help='Pickled model file to use (./models/*)')
        parser.add_argument('text', help='Text to score (or path to file)')
        args = parser.parse_args(args)

        model = load_model(args.model)

        with open(f'{args.model}.vocab', 'rb') as h:
            tokenizer = pickle.load(h)

        if os.path.exists(args.text):
            print(f'[+] Reading text from {args.text}')
            args.text = open(args.text).read()

        tokenized = tokenizer.texts_to_matrix([args.text])[0]
        prediction = model.predict(np.array([tokenized]))
        prediction = int(prediction[0][0] * 1000)

        print(f'\n[+] Predicted Score: {prediction}\n')

    def train(self, args):
        parser = argparse.ArgumentParser(
            prog=f'{sys.argv[0]} train',
            description='Train a new model using custom data')

        parser.add_argument('-d', '--data', required=True, choices=['links', 'texts'], help='Scored data type to use')
        parser.add_argument('output', help='Output path for model (.h5) file')
        parser.add_argument('--epochs', help='Training epochs', default=10)
        parser.add_argument('--batch_size', help='Batch size', default=64)
        parser.add_argument('--split_size', help='Test/train split ratio', default=0.2)
        args = parser.parse_args(args)
        
        if args.data == 'texts':
            scored_emails = dataloader.load_emails('core/data/texts.csv')
            text_matrix, score_matrix, tokenizer = preprocessing.tokenize_data(scored_emails, 'binary')

        elif args.data == 'links':
            scored_links = dataloader.load_links('core/data/links.csv')
            text_matrix, score_matrix, tokenizer = preprocessing.tokenize_links(scored_links, 'binary')

        print(f'[+] Training ...')

        x_train, x_test, y_train, y_test = train_test_split(text_matrix, score_matrix, test_size=args.split_size)
        model = models.create_neural_network(text_matrix.shape[1])

        early_stop = early_stop = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1)
        history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, callbacks=[early_stop])

        loss, mae = model.evaluate(x_test, y_test)
        print(f'[+] Mean score error: {int(mae * 1000)}')

        model.save(args.output)
        print(f'[+] Saved model to {args.output}')

        vocab_file = f'{args.output}.vocab'
        with open(vocab_file, 'wb') as h:
            pickle.dump(tokenizer, h, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[+] Saved vocab to {vocab_file}')

if __name__ == '__main__':
    ProofPudding(sys.argv[1:])