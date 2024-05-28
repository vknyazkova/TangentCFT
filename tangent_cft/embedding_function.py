import logging

from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
import numpy as np

from .tangent_cft_parser import TangentCFTParser
from .tangent_cft_back_end import TangentCFTBackEnd


class TangentCFTEmbedding(EmbeddingFunction[Documents]):
    def __init__(self,
                 tangent_cft: TangentCFTBackEnd,
                 mathml: bool = False,
                 slt: bool = False
                 ):
        self.tangent_cft = tangent_cft
        self.tangent_cft_parser = TangentCFTParser()
        self.mathml = mathml
        self.slt = slt

    def __call__(self, input: Documents) -> Embeddings:
        embeds = []
        for formula in tqdm(input):
            try:
                formula_tree_tuples = self.tangent_cft_parser.parse(formula, mathml=self.mathml, slt=self.slt)
                embeds.append(self.tangent_cft.get_formula_emebedding(formula_tree_tuples).tolist())
            except Exception as e:
                logging.debug(f'Problems with formula {formula}')
                embeds.append(np.random.random(300).tolist())
        return embeds

    