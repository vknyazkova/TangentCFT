from chromadb import Documents, EmbeddingFunction, Embeddings

from tangent_cft_back_end import TangentCFTBackEnd
from tangent_cft_encoder import FormulaTreeEncoder
from tangent_cft_module import TangentCFTModule
from tangent_cft_parser import TangentCFTParser
from Configuration.configuration import Configuration
from Embedding_Preprocessing.encoder_tuple_level import TupleTokenizationMode


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
        for formula in input:
            formula_tree_tuples = self.tangent_cft_parser.parse(formula, mathml=self.mathml, slt=self.slt)
            embeds.append(self.tangent_cft.get_formula_emebedding(formula_tree_tuples).tolist())
        return embeds


if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


    encoder = FormulaTreeEncoder.load('Models/Vocabulary/slt_encoder.tsv')

    ft_config = Configuration('Configuration/config/config_1')
    module = TangentCFTModule(ft_config, 'Models/FastText/slt_model.model')

    system = TangentCFTBackEnd(tangent_cft_module=module, encoder=encoder)
    embedder = TangentCFTEmbedding(tangent_cft=system, mathml=False, slt=True)

    latex_formulas = [
        '\\frac{\\partial \\mathbf{x}}{\\partial x}',
        '\\frac{x}{2}',
        '(x+2)^2'
    ]
    embeddings = embedder(latex_formulas)
    print(embeddings)
