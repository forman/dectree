import ast
import os.path
from typing import List, Optional, Tuple, Callable, Dict, Any

import click
import yaml
from decompiler import ExprDecompiler

BOOL_NAME = "FuzzyBool"


class Node(dict):
    def __init__(self):
        super().__init__()
        self.name = None
        self.type = None
        self.parent: Optional[Node] = None
        self.yes_children: List[Node] = []
        self.no_children: List[Node] = []


def parse_envi_dectree(envi_path: str) -> Tuple[List[Node], List[Node]]:
    with open(envi_path) as fp:
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
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            else:
                # noinspection PyBroadException
                try:
                    value = eval(value)
                except Exception:
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


def write_yaml_dectree(variables: List[Node],
                       nodes: List[Node],
                       config: Dict[str, Any],
                       write_line: Callable[[str], None] = print):
    rule_builder = RuleBuilder(config)
    rules = []
    for node in nodes:
        if node.parent is None:
            lines = rule_builder.build_rule(node, level=2)
            text = "\n".join(lines)
            rules.append(text)

    write_line("types:")
    for type_id, type_def in rule_builder.type_id_to_type_defs.items():
        write_line(f'  {type_id}:')
        for prop_name, prop_value in type_def.items():
            write_line(f'    {prop_name}: {prop_value}')

    write_line("")
    write_line("inputs:")
    for variable in variables:
        orig_var_id = variable["variable name"]
        var_id = config["variables"].get(orig_var_id)
        if var_id is None:
            print(f"warning: skipping variable {orig_var_id!r}")
            continue
        elif var_id not in rule_builder.used_vars:
            print(f"warning: unused variable {orig_var_id!r}"
                  f" (renamed to {var_id!r})")
            continue
        elif var_id in config["derived"]:
            continue
        type_name = rule_builder.var_id_to_type_ids.get(var_id, "float")
        file_name = variable.get("file name")
        if file_name is not None:
            # file_name = file_name.replace("\\", "\\\\")
            write_line(f'  # file name = "{file_name}"')
        file_pos = variable.get("file pos")
        if file_pos is not None:
            write_line(f'  # file pos = {file_pos}')
        write_line(f'  - {var_id}: {type_name}')

    if rule_builder.derived_vars or config["derived"]:
        write_line("")
        write_line("derived:")
        for var_id, var_expr in config["derived"].items():
            type_name = rule_builder.var_id_to_type_ids.get(var_id, "float")
            write_line(f'  - {var_id} = {var_expr}: {type_name}')
        for var_expr, var_id in rule_builder.derived_vars.items():
            type_name = rule_builder.var_id_to_type_ids.get(var_id, "float")
            write_line(f'  - {var_id} = {var_expr}: {type_name}')

    write_line("")
    write_line("outputs:")
    for var_id, expression in config["derived"].items():
        write_line(f'  - {var_id}: float')
    for node in nodes:
        if node["type"] == "Result":
            var_id = _name_to_id(node["name"])
            write_line(f'  - {var_id}: {BOOL_NAME}')

    write_line("")
    write_line("rules:")
    for rule in rules:
        write_line("  - |")
        write_line(rule)


def _name_to_id(name: str):
    if name.isidentifier():
        return name
    c = name[0]
    if c.isidentifier():
        id_ = ''
    else:
        id_ = 'var_'
    for c in name:
        if c.isalnum() or c == '_':
            pass
        elif c.isspace() or c == '-':
            c = '_'
        elif c == '<':
            c = 'lt'
        elif c == '>':
            c = 'gt'
        elif c == '=':
            c = 'eq'
        elif c == '!':
            c = 'not'
        else:
            c = '_'
        id_ += c
    return id_


class RuleBuilder(ExprDecompiler):
    op_names = {
        ast.Eq: "eq",
        ast.NotEq: "ne",
        ast.Gt: "gt",
        ast.GtE: "ge",
        ast.Lt: "lt",
        ast.LtE: "le",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.type_id_to_type_defs = {
            BOOL_NAME: {
                '"FALSE"': "false()",
                '"TRUE"': "true()",
            }
        }
        self.var_id_to_type_ids = {}
        self.derived_vars = {}
        self.used_vars = set()

    def transform_name(self, name: ast.Name):
        new_name = self.config["variables"].get(name.id)
        if new_name is None:
            print(f"warning: variable {name.id!r} will be used as-is")
            new_name = name.id
        self.used_vars.add(new_name)
        return new_name

    def transform_compare(self, left, ops, comparators):
        if len(ops) == 1:
            op = ops[0]
            right = comparators[0]
            if isinstance(left, ast.Name):
                # Left is a variable name
                var_expr = self.transform_name(left)
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
                if type_id in self.type_id_to_type_defs:
                    type_def = self.type_id_to_type_defs[type_id]
                else:
                    type_def = {}
                    self.type_id_to_type_defs[type_id] = type_def
                self.var_id_to_type_ids[var_id] = type_id
                type_def[prop_id] = f"{prop_func_id}({value})"
                return f"{var_id} is {prop_id}"

        return super().transform_compare(left, ops, comparators)

    def new_derived_var(self, var_expr):
        var_id = self.derived_vars.get(var_expr)
        if var_id is None:
            var_id = f"derived_{len(self.derived_vars)}"
            self.derived_vars[var_expr] = var_id
        self.used_vars.add(var_id)
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
            var_id = _name_to_id(node.get("name"))
            lines.append(f"{indent}{var_id} = TRUE")


@click.command()
@click.option('--out', '-o', 'out_path',
              help='Path to output file')
@click.option('--config', '-c', 'config_path',
              help='Path to configuration file')
@click.argument('envi_path')
def main(out_path: str = None,
         config_path: str = None,
         envi_path: str = None):
    config = {}
    if config_path is not None:
        with open(config_path) as file:
            config = yaml.safe_load(file)

    if out_path is None:
        out_dir = os.path.dirname(envi_path)
        out_name, _ = os.path.splitext(os.path.basename(envi_path))
        out_path = os.path.join(out_dir or ".", out_name + ".yaml")

    variables, nodes = parse_envi_dectree(envi_path)

    if "derived" not in config:
        config["derived"] = {}
    if not config.get("variables"):
        mapping = {}
        for variable in variables:
            var_name = variable.get('variable name')
            file_name = variable.get('file name')
            if file_name:
                new_name = os.path.splitext(os.path.basename(file_name))[0]
            else:
                new_name = var_name
            mapping[var_name] = new_name
            config["variables"] = mapping
        print("info: using following configuration:\n")
        print(yaml.dump(config))
        print()

    config["derived"] = {tuple(d.keys())[0]: tuple(d.values())[0]
                         for d in config["derived"]}

    with open(out_path, "w") as file:
        def write_line(line: str):
            file.write(line + "\n")

        write_yaml_dectree(variables, nodes, config, write_line=write_line)

    print(f"generated {out_path}")


if __name__ == "__main__":
    main()
