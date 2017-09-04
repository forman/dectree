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

Expression = str
DerivedDef = Tuple[VarName, TypeName, Expression]
DerivedDefs = List[DerivedDef]

RuleCondition = str
RuleIfKw = str
RuleElifKw = str
RuleElseKw = str
RuleAssignKw = str
RuleIf = Tuple[RuleIfKw, RuleCondition, "RuleBody"]
RuleElif = Tuple[RuleElifKw, RuleCondition, "RuleBody"]
RuleElse = Tuple[RuleElseKw, "RuleBody"]
RuleAssign = Tuple[RuleAssignKw, VarName, PropName]
RuleStmt = Union[RuleIf, RuleElif, RuleElse, RuleAssign]
RuleBody = List[RuleStmt]
Rules = List[RuleBody]