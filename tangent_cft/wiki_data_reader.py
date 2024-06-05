import os
from typing import Dict, List, Union
import unicodedata

from tqdm import tqdm

from .tangent_cft_parser import TangentCFTParser


class WikiDataReader:
    def __init__(self,
                 read_slt=True):
        self.read_slt = read_slt

    def get_collection(self,
                       collection_file_path,
                       tuples: bool = True) -> Dict[str, Union[str, List[str]]]:
        """
        This method read the NTCIR-12 formulae in the collection.
        To handle formulae with special characters line 39 normalizes the unicode data.
        The return value is a dictionary of formula id (as key) and list of tuples (as value)
        """
        except_count = 0
        formulas_tree_tuples = {}
        root = collection_file_path
        for directory in os.listdir(root):
            temp_address = root + "/" + directory + "/"
            if not os.path.isdir(temp_address):
                continue
            print(f"Processing {directory}")
            for filename in tqdm(os.listdir(temp_address)):
                file_path = temp_address + filename
                parts = filename.split('/')
                file_name = os.path.splitext(parts[len(parts) - 1])[0]
                temp = str(unicodedata.normalize('NFKD', file_name).encode('ascii', 'ignore'))
                temp = temp[2:]
                file_name = temp[:-1]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    mathml_formulas = TangentCFTParser.extract_mathml(content)
                    for i, mathml in enumerate(mathml_formulas):
                        if tuples:
                            parsed = TangentCFTParser.parse2tree_tuples(mathml, mathml=True, slt=self.read_slt)
                        else:
                            parsed = TangentCFTParser.parse2linearized_tree(mathml, mathml=True, slt=self.read_slt)
                        formulas_tree_tuples[file_name + ":" + str(i)] = parsed
                except Exception as e:
                    except_count += 1
        print(f'Number of exceptions during reading collection: {except_count}')
        return formulas_tree_tuples

    def get_query(self,
                  queries_directory_path,
                  tuples: bool = True) -> List[str]:
        """
        This method reads the NTCIR-12 the queries.
        Note that the Tangent-CFT does not support queries with Wildcard,
        Therefore the query range is 1 to 20 which are concerete queries in NTCIR-12.
        """
        except_count = 0
        formulas_tree_tuples = {}
        for j in range(1, 21):
            temp_address = queries_directory_path + '/' + str(j) + '.html'
            try:
                with open(temp_address, 'r', encoding='utf-8') as f:
                    content = f.read()
                mathml_formulas = TangentCFTParser.extract_mathml(content)
                for i, mathml in enumerate(mathml_formulas):
                    if tuples:
                        parsed = TangentCFTParser.parse2tree_tuples(mathml, mathml=True, slt=self.read_slt)
                    else:
                        parsed = TangentCFTParser.parse2linearized_tree(mathml, mathml=True, slt=self.read_slt)
                    formulas_tree_tuples[j] = parsed
            except Exception as e:
                print(e)
                except_count += 1
                print(j)
        return formulas_tree_tuples


if __name__ == '__main__':
    file_path = '../../NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles'
    queries = '../TestQueries'
    data_reader = WikiDataReader(read_slt=True)
    print(data_reader.get_query(queries))