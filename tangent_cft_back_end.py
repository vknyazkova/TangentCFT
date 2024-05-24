import logging
import os
import json
from typing import List, Dict, Union, Tuple

import numpy as np
from tqdm import tqdm

from Configuration.configuration import Configuration
from DataReader.abstract_data_reader import AbstractDataReader
from Embedding_Preprocessing.encoder_tuple_level import TupleEncoder, TupleTokenizationMode

from tangent_cft_module import TangentCFTModule


class TangentCFTBackEnd:
    def __init__(self,
                 ft_config: Configuration,
                 data_reader: AbstractDataReader,
                 embedding_type: TupleTokenizationMode,
                 ignore_full_relative_path: bool,
                 tokenize_all: bool,
                 tokenize_number: bool):
        """

        Args:
            ft_config: config for FastText model
            data_reader: data_reader
        """
        self.ft_config = ft_config
        self.data_reader = data_reader

        self.embedding_type = embedding_type
        self.ignore_full_relative_path = ignore_full_relative_path
        self.tokenize_all = tokenize_all
        self.tokenize_number = tokenize_number

        self.encoder_map_node = {}
        self.encoder_map_edge = {}
        self.node_id = 60000
        self.edge_id = 500
        self.module = None

    def __encode_tree_tuples(self,
                             tree_tuples: List[str]) -> List[str]:
        """
        Encode tree tuples. Each element of the tuple is tokenized according to config and encoded using encoder_map.
        (id from encoder map is transformed to unicode symbol in order to use it as a "character" in FastText model)
        Args:
            tree_tuples: OPT or SLT tree tuples (symbol 1, symbol 2, edge between them, full relative path from the root)
                which elements are joined by a tab
        Returns:
            tree tuples represented as sequence of characters
        """
        encoded_tuples, update_map_node, update_map_edge, node_id, edge_id = \
            TupleEncoder.encode_tuples(self.encoder_map_node, self.encoder_map_edge, self.node_id, self.edge_id,
                                       tree_tuples, self.embedding_type, self.ignore_full_relative_path, self.tokenize_all,
                                       self.tokenize_number)
        self.node_id = node_id
        self.edge_id = edge_id
        self.encoder_map_node.update(update_map_node)
        self.encoder_map_edge.update(update_map_edge)
        return encoded_tuples

    def __encode_train_data(self) -> Dict[str, List[str]]:
        """
        Iterates over train dataset and encodes each retrieved formula as encoded tree tuples
        Returns:
            {formula_id: [encoded tree tuple]}
        """
        logging.info("Reading train data...")
        dictionary_formula_slt_tuple = self.data_reader.get_collection()
        logging.info(f"Number of retrieved formulas: {len(dictionary_formula_slt_tuple.keys())}")

        logging.info("Encoding train data...")
        encoded_formulas = {}
        for formula in dictionary_formula_slt_tuple:
            encoded_formulas[formula] = self.__encode_tree_tuples(dictionary_formula_slt_tuple[formula])
        return encoded_formulas

    def train_model(self,
                    encoder_map_path: Union[os.PathLike, str],
                    ft_model_path: Union[os.PathLike, str],
                    encoded_train_formulas: Union[os.PathLike, str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a TangentCFT model.
        Args:
            encoder_map_path: path to encoder mapping file (or where it will be saved)
            ft_model_path: path where to save trained fasttext model
            encoded_train_formulas: path to encoded train formulas, if such file does not exist,
                then train formulas will be encoded and saved to this file
        Returns:
            embeddings (n_formulas, emb_size), formula_ids (n_formulas): vectorized train dataset with formula_ids
        """
        logging.info("Training TangentCFT model...")
        self.module = TangentCFTModule()
        if os.path.isfile(encoder_map_path):
            self.__load_encoder_map(encoder_map_path)

        if os.path.isfile(encoded_train_formulas):
            encoded_formulas = self.load_encoded_formulas(encoded_train_formulas)
        else:
            encoded_formulas = self.__encode_train_data()
            if encoded_train_formulas:
                self.save_encoded_formulas(encoded_formulas, encoded_train_formulas)

        self.__save_encoder_map(encoder_map_path)

        self.module.train_model(self.ft_config, list(encoded_formulas.values()))

        if ft_model_path is not None:
            self.module.save_model(ft_model_path)

        embeddings, formula_ids = self.module.index_collection(encoded_formulas)
        return embeddings, formula_ids

    def load_model(self,
                   encoder_map_path: Union[os.PathLike, str],
                   ft_model_path: Union[os.PathLike, str]):
        """
        Load the tangent-cft model from disk.
        Args:
            encoder_map_path: path to encoder mapping file
            ft_model_path: path to fasttext model
        """
        self.module = TangentCFTModule(ft_model_path)
        self.__load_encoder_map(encoder_map_path)

    def get_formula_emebedding(self,
                               formula_tree_tuples: List[str]) -> np.ndarray:
        encoded_tree_tuples = self.__encode_tree_tuples(formula_tree_tuples)
        formula_embedding = self.module.get_query_embedding(encoded_tree_tuples)
        return formula_embedding

    def retrieval(self,
                  dataset_embeddings: np.ndarray,
                  formula_ids: np.ndarray):
        logging.info("Formula Retrieval...")
        dictionary_query_tuples = self.data_reader.get_query()
        retrieval_result = {}
        for query, query_tuples in tqdm(dictionary_query_tuples.items()):
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

    def __save_encoder_map(self, map_file_path):
        """
        This method saves the encoder used for tokenization of formula tuples.
        map_file_path: file path to save teh encoder map in form of TSV file with column E/N \t character \t encoded value
        where E/N shows if the character is edge or node value, the character is tuple character to be encoded and encoded
        value is the value the encoder gave to character.
        """
        logging.info(f"Saving encoder map to {map_file_path}...")
        file = open(map_file_path, "w")
        for item in self.encoder_map_node:
            file.write("N" + "\t" + str(item) + "\t" + str(self.encoder_map_node[item]) + "\n")
        for item in self.encoder_map_edge:
            file.write("E" + "\t" + str(item) + "\t" + str(self.encoder_map_edge[item]) + "\n")
        file.close()

    def __load_encoder_map(self, map_file_path):
        """
        This method loads the saved encoder values into two dictionary used for edge and node values.
        """
        logging.info(f"Loading encoder_map from {map_file_path}...")
        file = open(map_file_path)
        line = file.readline().strip("\n")
        while line:
            parts = line.split("\t")
            encoder_type = parts[0]
            symbol = parts[1]
            value = int(parts[2])
            if encoder_type == "N":
                self.encoder_map_node[symbol] = value
            else:
                self.encoder_map_edge[symbol] = value
            line = file.readline().strip("\n")
        "The id shows the id that should be assigned to the next character to be encoded (a character that is not seen)" \
        "Therefore there is a plus one in the following lines"
        self.node_id = max(list(self.encoder_map_node.values())) + 1
        self.edge_id = max(list(self.encoder_map_edge.values())) + 1
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




