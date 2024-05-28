from pathlib import Path
from typing import List
from tqdm import tqdm
import unicodedata

from .TangentS.math_tan.math_extractor import MathExtractor
from .TangentS.math_tan.symbol_tree import SymbolTree


class TangentCFTParser:
    @staticmethod
    def extract_mathml(xml_content: str) -> List[str]:
        return MathExtractor.math_tokens(xml_content)

    @staticmethod
    def get_math_tree(formula: str, from_mathml: bool = True, slt: bool = True) -> SymbolTree:
        if not from_mathml:
            if slt:
                tree = MathExtractor.parse_from_tex(formula)
            else:

                tree = MathExtractor.parse_from_tex_opt(formula)
        else:
            if slt:
                slt_formula = MathExtractor.isolate_pmml(formula)
                tree = MathExtractor.convert_to_layoutsymbol(slt_formula)
            else:
                opt_formula = MathExtractor.isolate_cmml(formula)
                tree = MathExtractor.convert_to_semanticsymbol(opt_formula)
            tree = SymbolTree(tree)
        return tree

    @staticmethod
    def parse(formula: str, mathml: bool = True, slt: bool = True) -> List[str]:
        tree = TangentCFTParser.get_math_tree(formula, mathml, slt)
        return tree.get_pairs(window=2, eob=True)


def parse_ntcir_dataset(ntcir_math_folder: str, parsed_ntcir_folder: str):
    except_count = 0
    ntcir_math_folder = Path(ntcir_math_folder).resolve()
    parsed_ntcir_folder = Path(parsed_ntcir_folder).resolve()
    for directory in ntcir_math_folder.iterdir():
        if not directory.is_dir():
            continue
        print(f"Processing {directory}")
        dir_files = list(directory.iterdir())
        for filename in tqdm(dir_files):
            file_name = filename.stem
            temp = str(unicodedata.normalize('NFKD', file_name).encode('ascii', 'ignore'))
            file_name = temp[2:-1]
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                mathml_formulas = TangentCFTParser.extract_mathml(content)
                for i, mathml in enumerate(mathml_formulas):
                    new_file = Path(parsed_ntcir_folder, f'{file_name}_{str(i)}').with_suffix('.html')
                    with open(new_file, 'w', encoding='utf-8') as f:
                        f.write(mathml)
            except Exception as e:
                except_count += 1
    print(f'Number of exceptions during reading collection: {except_count}')
    return parsed_ntcir_folder


if __name__ == '__main__':
    parser = TangentCFTParser()
    latex_formulas = [
        '\\frac{\\partial \\mathbf{x}}{\\partial x}',
        '\\frac{x}{2}',
        '(x+2)^2'
    ]
    mathml_example = '''
<math display="inline" id="dash-yllion:0">
 <semantics>
  <msup>
   <mn>10</mn>
   <msup>
    <mn>2</mn>
    <mrow>
     <mi>n</mi>
     <mo>+</mo>
     <mn>2</mn>
    </mrow>
   </msup>
  </msup>
  <annotation-xml encoding="MathML-Content">
   <apply>
    <csymbol cd="ambiguous">superscript</csymbol>
    <cn type="integer">10</cn>
    <apply>
     <csymbol cd="ambiguous">superscript</csymbol>
     <cn type="integer">2</cn>
     <apply>
      <plus></plus>
      <ci>n</ci>
      <cn type="integer">2</cn>
     </apply>
    </apply>
   </apply>
  </annotation-xml>
  <annotation encoding="application/x-tex">
   10^{2^{n+2}}
  </annotation>
 </semantics>
</math>'''
    print(parser.parse(latex_formulas[0], mathml=False, slt=False))
    print(parser.parse(latex_formulas[0], mathml=False, slt=True))
    print(parser.parse(mathml_example, mathml=True, slt=False))
    print(parser.parse(mathml_example, mathml=True, slt=True))
#
#     example_doc_path = '/Users/viktoriaknazkova/Desktop/me/study/github_repos/NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles/wpmath0000001/Geometric_algebra.html'
#     with open(example_doc_path, 'r') as f:
#         example_doc = f.read()
#     print(len(parser.extract_mathml(example_doc)))
