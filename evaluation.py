from typing import Optional, List, Dict

import pandas as pd
from chromadb import Client, Documents, EmbeddingFunction, Embeddings

from tangent_cft_encoder import FormulaTreeEncoder
from tangent_cft_module import TangentCFTModule
from tangent_cft_back_end import TangentCFTBackEnd
from Configuration.configuration import Configuration
from embedding_functions import TangentCFTEmbedding


dataset = {
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


def index_dataset(chroma_client: Client,
                  collection_name: str,
                  embedding_fn: EmbeddingFunction,
                  documents: Documents,
                  ids: List[str],
                  embeddings: Optional[Embeddings] = None):
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn,
                                                        metadata={"hnsw:space": "cosine"})
    collection.upsert(
        documents=documents,
        ids=ids,
        embeddings=embeddings,
        )


def run_test_queries(chroma_client: Client,
                 collection_name: str,
                 embedding_fn: EmbeddingFunction,
                 queries: Dict[str, str],
                 queries_embeds: Optional[Embeddings] = None,
                 n_results: int = 10):
    collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_fn)
    res = collection.query(query_embeddings=queries_embeds, query_texts=list(queries.values()), n_results=n_results, include=['distances', 'documents'])
    results = []
    for i, (query_id, query) in enumerate(queries.items()):
        for j in range(n_results):
            results.append([
                query_id,
                query,
                j + 1,
                res['documents'][i][j]
            ])
    return pd.DataFrame(results, columns=['query_id', 'query', 'rank', 'formula'])


if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    encoder = FormulaTreeEncoder.load('Models/Vocabulary/slt_type_encoder.tsv')
    ft_config = Configuration('Configuration/config/config_3')
    module = TangentCFTModule(ft_config, 'Models/FastText/slt_type_model.model')
    system = TangentCFTBackEnd(tangent_cft_module=module, encoder=encoder)

    chroma_client = Client()
    tangent_cft_embedder = TangentCFTEmbedding(tangent_cft=system, mathml=False, slt=True)
    collection_name = 'my_test_slt_type'
    index_dataset(chroma_client, collection_name=collection_name, embedding_fn=tangent_cft_embedder,
                  documents=list(dataset.values()), ids=list(dataset.keys()), embeddings=None)
    retrieved = run_test_queries(chroma_client, collection_name, tangent_cft_embedder, queries=queries, n_results=10)
    retrieved.to_csv('/Users/viktoriaknazkova/Desktop/me/study/thesis/my_test_slt_type.tsv', header=True, index=False, sep='\t')