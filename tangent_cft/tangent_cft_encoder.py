from typing import Dict, List
import logging
from tqdm import tqdm

from .Embedding_Preprocessing.encoder_tuple_level import TupleEncoder, TupleTokenizationMode


class FormulaTreeEncoder:
    def __init__(self,
                 embedding_type: TupleTokenizationMode = TupleTokenizationMode.Both_Separated,
                 ignore_full_relative_path: bool = True,
                 tokenize_all: bool = False,
                 tokenize_number: bool = True,
                 node_vocabulary: Dict[str, int] = None,
                 edge_vocabulary: Dict[str, int] = None,
                 node_id: int = 6000,
                 edge_id: int = 500):

        self.embedding_type = embedding_type
        self.ignore_full_relative_path = ignore_full_relative_path
        self.tokenize_all = tokenize_all
        self.tokenize_number = tokenize_number

        self.node_vocabulary = node_vocabulary if node_vocabulary is not None else {}
        self.edge_vocabulary = edge_vocabulary if edge_vocabulary is not None else {}
        self.node_id = node_id
        self.edge_id = edge_id

    def __encode_tuples(self,
                        formula_tuples: List[str]) -> List[str]:
        encoded_tuples, update_map_node, update_map_edge, node_id, edge_id = TupleEncoder.encode_tuples(
            self.node_vocabulary,
            self.edge_vocabulary,
            self.node_id,
            self.edge_id,
            formula_tuples,
            self.embedding_type,
            self.ignore_full_relative_path,
            self.tokenize_all,
            self.tokenize_number
        )
        self.node_id = node_id
        self.edge_id = edge_id
        self.node_vocabulary.update(update_map_node)
        self.edge_vocabulary.update(update_map_edge)
        return encoded_tuples

    def fit_transform(self,
                      formula_tree_tuples: List[List[str]],
                      verbose: bool = False) -> List[List[str]]:
        encoded = []
        for formula in tqdm(formula_tree_tuples, disable=not verbose):
            encoded_tuples = self.__encode_tuples(formula)
            encoded.append(encoded_tuples)
        return encoded

    def transform(self,
                  formula_tree_tuples: List[List[str]],
                  verbose: bool = False) -> List[List[str]]:
        encoded = []
        for formula in tqdm(formula_tree_tuples, disable=not verbose):
            encoded_tuples = self.__encode_tuples(formula)
            encoded.append(encoded_tuples)
        return encoded

    def save_vocabulary(self, vocabulary_map_path):
        """
        This method saves the encoder used for tokenization of formula tuples.
        map_file_path: file path to save teh encoder map in form of TSV file with column E/N \t character \t encoded value
        where E/N shows if the character is edge or node value, the character is tuple character to be encoded and encoded
        value is the value the encoder gave to character.
        """
        logging.info(f"Saving vocabulary to {vocabulary_map_path}...")
        with open(vocabulary_map_path, "w", encoding="utf-8") as file:
            file.write(f'Emb_type={str(self.embedding_type.value)}\tIFRP={str(self.ignore_full_relative_path)}\t'
                       f'Tokenize_all={str(self.tokenize_all)}\tTokenize_number={str(self.tokenize_number)}\n')
            for item in self.node_vocabulary:
                file.write("N" + "\t" + str(item) + "\t" + str(self.node_vocabulary[item]) + "\n")
            for item in self.edge_vocabulary:
                file.write("E" + "\t" + str(item) + "\t" + str(self.edge_vocabulary[item]) + "\n")

    @classmethod
    def load(cls, vocabulary_map_path):
        logging.info(f"Loading vocabulary from {vocabulary_map_path}...")
        with open(vocabulary_map_path, "r", encoding="utf-8") as file:
            params = file.readline().strip("\n").split("\t")
            print(params)
            params = {el.split('=')[0]: el.split('=')[1] for el in params}
            print(params)
            embedding_type = TupleTokenizationMode(int(params['Emb_type']))
            ignore_full_relative_path = bool(params['IFRP'])
            tokenize_all = bool(params['Tokenize_all'])
            tokenize_number = bool(params['Tokenize_number'])

            line = file.readline().strip("\n")
            node_vocabulary = {}
            edge_vocabulary = {}
            while line:
                parts = line.split("\t")
                encoder_type = parts[0]
                symbol = parts[1]
                value = int(parts[2])
                if encoder_type == "N":
                    node_vocabulary[symbol] = value
                else:
                    edge_vocabulary[symbol] = value
                line = file.readline().strip("\n")

            "The id shows the id that should be assigned to the next character to be encoded (a character that is not seen)" \
            "Therefore there is a plus one in the following lines"
            node_id = max(list(node_vocabulary.values())) + 1
            edge_id = max(list(edge_vocabulary.values())) + 1

            return cls(
                embedding_type=embedding_type,
                ignore_full_relative_path=ignore_full_relative_path,
                tokenize_all=tokenize_all,
                tokenize_number=tokenize_number,
                node_vocabulary=node_vocabulary,
                edge_vocabulary=edge_vocabulary,
                node_id=node_id,
                edge_id=edge_id
            )
