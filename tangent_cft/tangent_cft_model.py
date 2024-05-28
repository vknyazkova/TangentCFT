import os
from typing import List, Union
import logging

import numpy as np
from gensim.models import FastText

from .configuration import Configuration


class TangentCftModel:
    def __init__(self,):
        self.model = None

    def train(self,
              config: Configuration,
              fast_text_train_data: List[List[str]]):
        """
        Training fasttext model on training data, where one "sentence" is a formula and
            one "word" is a tree tuple, encoded as string, and "characters" are tree tuple elements.
        Args:
            config: Configuration object with fasttext training parameters
            fast_text_train_data: List[List[encoded_tuple]]
        """

        size = config.vector_size
        window = int(config.context_window_size)
        sg = int(config.skip_gram)
        hs = int(config.hs)
        negative = int(config.negative)
        iteration = int(config.iter)
        min_n = int(config.min)
        max_n = int(config.max)
        word_ngrams = int(config.ngram)

        logging.info("Training FastText model")
        self.model = FastText(fast_text_train_data, vector_size=size, window=window, sg=sg, hs=hs,
                              workers=1, negative=negative, epochs=iteration, min_n=min_n,
                              max_n=max_n, word_ngrams=word_ngrams)

    def save_model(self,
                   model_file_path: Union[str, os.PathLike]):
        logging.info(f"Saving the fast text model to {model_file_path}")
        self.model.save(model_file_path)

    def load_model(self,
                   model_file_path: Union[str, os.PathLike]):
        self.model = FastText.load(model_file_path)

    def get_vector_representation(self,
                                  encoded_tree_tuple: str) -> np.ndarray:
        """
        Get vector representation of a tree tuple.
        Args:
            encoded_tree_tuple: tree tuple encoded as string

        Returns:
            tree_tuple_embedding: (emb_size)
        """
        return self.model.wv[encoded_tree_tuple]
