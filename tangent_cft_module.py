import os
from typing import List, Union, Dict, Tuple
import logging

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from Configuration.configuration import Configuration
from tangent_cft_model import TangentCftModel


class TangentCFTModule:
    def __init__(self,
                 ft_model_path: Union[os.PathLike, str] = None):

        self.model = TangentCftModel()
        if ft_model_path is not None:
            logging.info("Loading pre-trained model from {}".format(ft_model_path))
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
            except KeyError:
                logging.debug(f"Key Error for {encoded_tuple}")
        if len(tuple_embeddings) == 0:
            raise KeyError("Any formula tuple has no embeddings")
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
        logging.info("Indexing dataset...")
        embeddings = []
        formula_ids = []
        idx = 0
        for formula_id in tqdm(encoded_formulas):
            try:
                formula_embedding = self.__get_formula_embedding(encoded_formulas[formula_id])
                embeddings.append(formula_embedding)
                formula_ids.append(formula_id)
                idx += 1
            except Exception as e:
                logging.debug(f'For formula {formula_id}\n', exc_info=e)
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
            top_n: int = 1000) -> Dict[str, float]:
        """

        Args:
            collection_embeddings: formulas embeddings (n_formulas, emb_size)
            formulas_ids: formula_ids (n_formulas)
            query_embedding: query formula embedding (emb_size)
            top_n: top n results to retrieve

        Returns:
            mapping from formula to its score
        """
        scores = (collection_embeddings @ query_embedding.reshape(300, 1)) / (
                norm(collection_embeddings, axis=1, keepdims=True) * norm(query_embedding, keepdims=True)
        )
        rank = (-scores).argsort(axis=0)[:, 0][:top_n]
        selected_formulas_ids = formulas_ids[rank]
        selected_formulas_scores = scores[rank]
        return dict(zip(selected_formulas_ids.tolist(), selected_formulas_scores.reshape(-1,).tolist()))
