from typing import List

from TangentS.math_tan.math_extractor import MathExtractor
from TangentS.math_tan.symbol_tree import SymbolTree


class TangentCFTParser:
    @staticmethod
    def parse(formula: str, mathml: bool = True, slt: bool = True) -> List[str]:
        if not mathml:
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
        return tree.get_pairs(window=2, eob=True)


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