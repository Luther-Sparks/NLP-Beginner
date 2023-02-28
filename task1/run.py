import numpy
import csv
import random
import sys
from feature import BoW, N_Gram
from show import alpha_gradient_plot

if __name__ == '__main__':
    with open('./sentiment-analysis-on-movie-reviews/train.tsv') as f:
        # tsvreader = csv.reader(f, delimiter)
        all_data = list(csv.reader(f, delimiter='\t'))

    # Initialization
    data = all_data[1:]
    max_item = 1000

    # Feature extraction
    bow = BoW(data, max_item)
    bow.get_words()
    bow.get_matrix()

    # Feature extraction
    n_gram = N_Gram(data, 3, max_item)
    n_gram.get_words()
    n_gram.get_matrix()

    # Show plot
    alpha_gradient_plot(bow, n_gram, 10000, 10)
    alpha_gradient_plot(bow, n_gram, 100000, 10)