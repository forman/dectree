import ast
import sys
from typing import List, Optional, Tuple

from decompiler import ExprDecompiler


class Node(dict):
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.parent: Optional[Node] = None
        self.yes_children: List[Node] = []
        self.no_children: List[Node] = []


def parse_envi_dectree(envi_file: str) -> Tuple[List[Node], List[Node]]:
    with open(envi_file) as fp:
        lines = fp.readlines()
    nodes = []
    variables = []
    current_node = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("ENVI Decision Tree"):
            continue
        if line == 'begin node':
            current_node = Node()
        elif line == 'end node':
            nodes.append(current_node)
            current_node = None
        elif line == 'begin variable':
            current_node = Node()
        elif line == 'end variable':
            variables.append(current_node)
            current_node = None
        elif current_node is not None:
            if current_node is None:
                raise ValueError(f"syntax error: {line}")
            key, value = [token.strip()
                          for token in line.split("=", maxsplit=2)]
            try:
                value = eval(value)
            except NameError:
                pass
            current_node[key] = value

    decisions = {node["name"]: node for node in nodes
                 if node["type"] == "Decision"}
    for node in nodes:
        if "parent name" in node:
            parent_name = node["parent name"]
            parent = decisions[parent_name]
            node.parent = parent
            decision = node.get("parent decision")
            if decision == "Yes":
                parent.yes_children.append(node)
            elif decision == "No":
                parent.no_children.append(node)
            else:
                raise ValueError("Syntax error")

    return variables, nodes


def write_yaml_dectree(variables: List[Node], nodes: List[Node]):
    rule_builder = RuleBuilder()
    rules = []
    for node in nodes:
        if node.parent is None:
            lines = rule_builder.build_rule(node, level=2)
            text = "\n".join(lines)
            rules.append(text)

    print("types:")
    for type_name in rule_builder.type_names.values():
        print(f'  {type_name}:')
        type_def = rule_builder.type_defs[type_name]
        for prop_name, prop_value in type_def.items():
            print(f'    {prop_name}: {prop_value}')

    print("")
    print("inputs:")
    for variable in variables:
        var_id = variable["variable name"]
        type_name = rule_builder.type_names.get(var_id, "float")
        print(f'  - {var_id}: {type_name}')

    print("")
    print("outputs:")
    for node in nodes:
        if node["type"] == "Result":
            print(f'  - {node["name"]}: Bool')

    if rule_builder.derived_vars:
        print("")
        print("derived:")
        for var_expr, var_id in rule_builder.derived_vars.items():
            type_name = rule_builder.type_names.get(var_id, "float")
            print(f'  - {var_id} = {var_expr}: {type_name}')

    print("")
    print("rules:")
    for rule in rules:
        print("  - |")
        print(rule)


class RuleBuilder(ExprDecompiler):
    op_names = {
        ast.Eq: "eq",
        ast.NotEq: "ne",
        ast.Gt:"gt",
        ast.GtE: "ge",
        ast.Lt: "lt",
        ast.LtE: "le",
    }

    def __init__(self):
        self.type_defs = {}
        self.type_names = {}
        self.derived_vars = {}

    def transform_compare(self, left, ops, comparators):
        if len(ops) == 1:
            op = ops[0]
            right = comparators[0]
            if isinstance(left, ast.Name):
                # Left is a variable name
                var_expr = left.id
                var_id = var_expr
            else:
                # Left is another expression: create a derived variable
                # from expression and use its new name instead
                var_expr = self.decompile(left)
                var_id = self.new_derived_var(var_expr)

            if isinstance(right, ast.Constant):
                # Right is constant
                value = right.value
                value_id = f'{value}'.replace(".", "")
            else:
                # Right is another expression: create a derived variable
                # using left expression minus right expression
                # and compare with zero
                value = 0
                value_id = "0"
                right_expr = self.decompile(right)
                # Note: we could avoid parentheses
                var_expr = f"{var_expr} - ({right_expr})"
                var_id = self.new_derived_var(var_expr)

            op_type = type(op)
            if op_type in self.op_names:
                prop_func_id = self.op_names[op_type]
                type_id = var_id.upper()
                prop_id = f"{prop_func_id.upper()}_{value_id}"
                if type_id in self.type_defs:
                    type_def = self.type_defs[type_id]
                else:
                    type_def = {}
                    self.type_defs[type_id] = type_def
                self.type_names[var_id] = type_id
                type_def[prop_id] = f"{prop_func_id}({value})"
                return f"{var_id} is {prop_id}"

        return super().transform_compare(left, ops, comparators)

    def new_derived_var(self, var_expr):
        var_id = self.derived_vars.get(var_expr)
        if var_id is None:
            var_id = f"temp_{len(self.derived_vars)}"
            self.derived_vars[var_expr] = var_id
        return var_id

    def build_rule(self, node: Node,  level: int = 0) -> List[str]:
        lines = []
        self._build_rule(node, level, lines)
        return lines

    def _build_rule(self,
                    node: Node,
                    level: int,
                    lines: List[str]):
        indent = level * "  "
        expression = node.get('expression')
        if expression:
            python_expr = expression.strip()
            for a, b in (("eq", "=="),
                         ("ne", "!="),
                         ("gt", ">"),
                         ("ge", ">="),
                         ("lt", "<"),
                         ("le", "<=")):
                python_expr = python_expr.replace(f" {a} ", f" {b} ")
            expr_node = ast.parse(python_expr)
            expression = self.decompile(expr_node)
            lines.append(f"{indent}if {expression}:")
            assert len(node.yes_children) > 0
            for yes_node in node.yes_children:
                self._build_rule(yes_node, level + 1, lines)
            if node.no_children:
                lines.append(f"{indent}else:")
                for no_node in node.no_children:
                    self._build_rule(no_node, level + 1, lines)
        else:
            name = node.get("name")
            lines.append(f"{indent}{name} = TRUE")


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    envi_file = args[0]
    variables, nodes = parse_envi_dectree(envi_file)
    write_yaml_dectree(variables, nodes)


if __name__ == "__main__":
    main()
