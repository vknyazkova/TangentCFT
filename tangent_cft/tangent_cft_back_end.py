import logging
import os
import json
from typing import List, Dict, Union

import numpy as np
from tqdm import tqdm

from .tangent_cft_module import TangentCFTModule
from .tangent_cft_encoder import FormulaTreeEncoder
from .configuration import Configuration


class TangentCFTBackEnd:
    def __init__(self,
                 tangent_cft_module: TangentCFTModule,
                 encoder: FormulaTreeEncoder):
        self.encoder = encoder
        self.module = tangent_cft_module

    @classmethod
    def load(cls,
             encoder_map_path: str,
             ft_model_path: str):
        encoder = FormulaTreeEncoder.load(encoder_map_path)
        module = TangentCFTModule(ft_model_path)
        return cls(tangent_cft_module=module, encoder=encoder)

    def train_model(self,
                    ft_config: Configuration,
                    train_formula_tree_tuples: List[List[str]] = None,
                    encoded: bool = False):
        if not encoded:
            logging.info("Encoding train data...")
            encoded_formulas = self.encoder.fit_transform(train_formula_tree_tuples)
        else:
            encoded_formulas = train_formula_tree_tuples
        self.module.train_model(ft_config, encoded_formulas)

    def save_model(self,
                   ft_model_path: Union[os.PathLike, str],
                   vocabulary_map_path: Union[os.PathLike, str]):
        self.module.save_model(ft_model_path)
        self.encoder.save_vocabulary(vocabulary_map_path)

    def get_formula_emebedding(self,
                               formula_tree_tuples: List[str]) -> np.ndarray:
        encoded_tree_tuples = self.encoder.transform([formula_tree_tuples])[0]
        formula_embedding = self.module.get_formula_embedding(encoded_tree_tuples)
        return formula_embedding

    def retrieval(self,
                  dataset_embeddings: np.ndarray,
                  formula_ids: np.ndarray,
                  query_fromula_tree_tuples: Dict[str, List[str]]):
        logging.info("Formula Retrieval...")
        retrieval_result = {}
        for query, query_tuples in tqdm(query_fromula_tree_tuples.items()):
            query_vec = self.get_formula_emebedding(query_tuples)
            retrieval_result[query] = self.module.formula_retrieval(dataset_embeddings, formula_ids, query_vec)
        return retrieval_result

    @staticmethod
    def create_result_file(
            result_query_doc: Dict[str, Dict[str, float]],
            result_file_path: Union[str, os.PathLike],
            run_id: int):
        """
        Creates result files in Trec format that can be used for trec_eval tool
        """
        logging.info(f"Saving retrieval result to {result_file_path}...")
        file = open(result_file_path, "w")
        for query_id in result_query_doc:
            count = 1
            query = "NTCIR12-MathWiki-" + str(query_id)
            line = query + " xxx "
            for doc_id, score in result_query_doc[query_id].items():
                temp = line + doc_id + " " + str(count) + " " + str(score) + " Run_" + str(run_id)
                count += 1
                file.write(temp + "\n")
        file.close()

    @staticmethod
    def save_encoded_formulas(encoded_formulas: Dict[str, List[str]],
                              path_to_save: Union[str, os.PathLike]):
        """
        Saves the encoded formulas to the given path.
        Args:
            encoded_formulas: {formula_id: List[encoded tree tuple]}
            path_to_save: where to save the encoded formulas
        """
        logging.info(f"Saving encoded train formulas to: {path_to_save}")
        with open(path_to_save, 'w', encoding='utf-8') as f:
            json.dump(encoded_formulas, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_encoded_formulas(encoded_formulas_path: Union[str, os.PathLike]) -> Dict[str, List[str]]:
        logging.info(f"Loading encoded formulas from {encoded_formulas_path}...")
        with open(encoded_formulas_path, 'r', encoding='utf-8') as f:
            encoded_formulas = json.load(f)
        return encoded_formulas




