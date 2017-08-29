from typing import Tuple, Dict, Any, List, Union

PropName = str
PropValue = str
PropFuncParamName = str
PropFuncParamValue = Any
PropFuncParams = Dict[PropFuncParamName, PropFuncParamValue]
PropFuncBody = str
PropFuncResult = Tuple[PropFuncParams, PropFuncBody]
PropDef = Tuple[PropValue, PropFuncParams, PropFuncBody]

TypeName = str
TypeDef = Dict[PropName, PropDef]
TypeDefs = Dict[TypeName, TypeDef]

VarName = str
VarDefs = Dict[VarName, TypeName]

InputDefs = VarDefs
OutputDefs = VarDefs

Rule = str
Rules = List[Rule]
RuleOrRules = Union[Rule, Rules]
