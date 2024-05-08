import logging
import os
from typing import List, Union, Dict, Tuple

import numpy as np
import torch
from numpy.linalg import norm

from Configuration.configuration import Configuration
from tangent_cft_model2 import TangentCftModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TangentCFTModule:
    def __init__(self,
                 ft_model_path: Union[os.PathLike, str] = None):

        self.model = TangentCftModel()
        if ft_model_path is not None:
            print("Loading pre-trained model from {}".format(ft_model_path))
            self.model.load_model(ft_model_path)

    def train_model(self,
                    configuration: Configuration,
                    encoded_formulas: List[List[str]]) -> None:
        """
        Train a TangentCftModel
        Args:
            configuration: Configuration object with fasttext training parameters
            encoded_formulas: list of formulas, each represented as a list of encoded tree tuples
        """
        self.model.train(configuration, encoded_formulas)

    def save_model(self, ft_model_path: Union[os.PathLike, str]) -> None:
        self.model.save_model(ft_model_path)

    def __get_formula_embedding(self,
                                encoded_formula: List[str]) -> np.ndarray:
        """
        Converts encoded formula tuples to formula embeddings.
        Args:
            encoded_formula: list of encoded tree tuples
        Returns:
            formula_embedding: (emb_size)
        """
        tuple_embeddings = []
        for encoded_tuple in encoded_formula:
            try:
                emb = self.model.get_vector_representation(encoded_tuple)
                tuple_embeddings.append(emb)
            except KeyError as e:
                logging.exception(e)
        return np.array(tuple_embeddings).mean(axis=0)

    def index_collection(self,
                         encoded_formulas: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Index collection using TangentCftModel
        Args:
            encoded_formulas: dictionary of formula_id: encoded formula

        Returns:
            index, formula_ids: formula embeddings (n_formulas, emb_size), formulas ids (n_formulas)
        """
        embeddings = []
        formula_ids = []
        idx = 0
        for formula_id in encoded_formulas:
            try:
                formula_embedding = self.__get_formula_embedding(encoded_formulas[formula_id])
                embeddings.append(formula_embedding)
                formula_ids.append(formula_id)
                idx += 1
            except Exception as e:
                logging.exception(e)
                continue
        embeddings = np.array(embeddings)
        formula_ids = np.array(formula_ids)
        return embeddings, formula_ids

    def get_query_embedding(self,
                            encoded_formula: List[str]) -> np.ndarray:
        """
        Get query embedding using TangentCftModel
        Args:
            encoded_formula: list of encoded tree tuples

        Returns:
            formula embedding: (emb_size)
        """
        return self.__get_formula_embedding(encoded_formula)

    @staticmethod
    def formula_retrieval(
            collection_embeddings: np.ndarray,
            formulas_ids: np.ndarray,
            query_embedding: np.ndarray,
            top_n: int = 1000):
        """

        Args:
            collection_embeddings:
            formulas_ids:
            query_embedding:
            top_n:

        Returns:

        """
        scores = (collection_embeddings @ query_embedding.reshape(300, 1)) / (
                norm(collection_embeddings, axis=1, keepdims=True) * norm(query_embedding, keepdims=True)
        )
        rank = (-scores).argsort(axis=0)[:, 0][:top_n]
        selected_formulas_ids = formulas_ids[rank]
        selected_formulas_scores = scores[rank]
        return dict(zip(selected_formulas_ids, selected_formulas_scores))
