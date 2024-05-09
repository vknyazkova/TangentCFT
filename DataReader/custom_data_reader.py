from typing import Dict, List

from DataReader.abstract_data_reader import AbstractDataReader
from TangentS.Tuple_Extraction import latex_math_to_slt_tuples, latex_math_to_opt_tuples


class LatexDataReader(AbstractDataReader):
    def __init__(self, latex_formulas: List[str] = None, read_slt=True):
        self.latex_formulas = latex_formulas
        self.read_slt = read_slt
        ...

    def get_collection(self) -> Dict[str, List[str]]:
        dictionary_formula_tuples = {}
        except_count = 0
        for formula_id, latex_string in enumerate(self.latex_formulas):
            try:
                if self.read_slt:
                    lst_tuples = latex_math_to_slt_tuples(latex_string)
                else:
                    lst_tuples = latex_math_to_opt_tuples(latex_string)
                dictionary_formula_tuples[str(formula_id)] = lst_tuples
            except Exception as e:
                except_count += 1
        print(f'Number of exceptions during reading collection: {except_count}')
        return dictionary_formula_tuples

    def get_query(self):
        ...


if __name__ == '__main__':
    latex_formulas = [
        '\\frac{\\partial \\mathbf{x}}{\\partial x}',
        '\\frac{x}{2}',
        '(x+2)^2'
    ]
    print(LatexDataReader(latex_formulas, read_slt=False).get_collection())
