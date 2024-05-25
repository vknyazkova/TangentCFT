from chromadb import Documents, EmbeddingFunction, Embeddings, Client

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


    encoder = FormulaTreeEncoder.load('Models/Vocabulary/opt_encoder.tsv')

    ft_config = Configuration('Configuration/config/config_1')
    module = TangentCFTModule(ft_config, 'Models/FastText/opt_model.model')

    system = TangentCFTBackEnd(tangent_cft_module=module, encoder=encoder)
    embedder = TangentCFTEmbedding(tangent_cft=system, mathml=False, slt=True)

    qualitive_dataset = {
        '1:a': '(a+b)\\times(c+d)',
        '1:b': '\\frac{1+2}{3+4}',
        '1:c': '(a+b)/(c+d)',
        '1:d': '(a+b) \\div (c+d)',
        '1:e': '(1+2) \\times (3+4)',
        '1:f': '\\frac{a+b}{c}',
        '1:g': '\\frac{a+b}{c+d}',
        '2:a': 'a*b',
        '2:b': 'a\\times b',
        '2:c': 'a \\cdot b',
        '2:d': 'ab',
        '2:e': '5\\cdot 6',
        '2:f': 'a \\div b',
        '3:a': '\\int_a^b h(x) dx',
        '3:b': '\\int_a^b v(t) dt',
        '3:c': '\\int g(x) dx',
        '3:d': '\\int_{-2}^5 g(x)dx',
        '3:e': 'g(x)',
        '3:f': '\\int_a^b g(x) dx',
        '4:a': '\\int_0^1 g(t) dt',
        '4:b': '\\sum_{i=1}^n a_i',
        '4:c': '\\frac{d}{dx} f(x)',
        '5:a': 'E = \\frac{1}{2} mv^2',
        '5:b': 'E = hv',
        '5:c': 'F = ma',
        '5:d': 'pV = nRT',
        '6:a': 'e^{ix} = cos(x) + i\\ sin(x)',
        '6:b': 'e^x = \\sum_{n=0}^{\infin} \\frac{x^n}{n!}',
        '6:c': 'ln(e) = 1'
    }
    queries = {
        '1': '\\frac{a+b}{c+d}',
        '2': 'ab',
        '3': '\\int_a^b g(x) dx',
        '4': '\\int_a^b f(x) dx',
        '5': 'E=mc^2',
        '6': 'e^{i\\pi} + 1 = 0',
    }

    chroma_client = Client()
    collection = chroma_client.create_collection(name="my_collection", embedding_function=embedder)

    collection.add(
        documents=list(qualitive_dataset.values()),
        ids=list(qualitive_dataset.keys())
    )
    results = collection.query(query_texts=list(queries.values()))
    print(results['documents'])