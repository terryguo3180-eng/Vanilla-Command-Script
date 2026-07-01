from vcs import ast
from vcs import lexer as lex
from vcs import parser as psr
from vcs import utils


class ConstantFolder(ast.ASTNodeTransformer):
    def __init__(self, parser: psr.Parser, dump_cf=False):
        self.parser = parser
        self.dump_cf = dump_cf
    
    def fold(self):
        tree = self.parser.parse()
        folded = self.visit(tree)
        if self.dump_cf:
            utils.print_info(utils.dump_astnode(folded))
        return folded
    
    def get_error_info_on(self, node: ast.ASTNode | lex.TokenInfo):
        return self.parser.get_error_info_on(node)

    def _apply_loc(self, source: ast.ASTNode, target: ast.ASTNode):
        new = target.copy()
        new.lineno = source.lineno
        new.column = source.column
        new.end_lineno = source.end_lineno
        new.end_column = source.end_column
        return new

    def _check_bool(self, node: ast.Constant):
        return isinstance(node.type, ast.IntType) or isinstance(node.type, ast.BoolType)

    def _check_true(self, node: ast.Constant):
        return (
            isinstance(node.type, ast.IntType) and node.value != 0
            or isinstance(node.type, ast.BoolType) and node.value is True
        )

    def visit_IfStatement(self, node: ast.IfStatement):
        test = self.visit(node.test)
        
        if not isinstance(test, ast.Constant):
            return node
        if not self._check_bool(test):
            return node

        if self._check_true(test):
            return self._apply_loc(node, node.body)
        
        return node.orelse

    def _handle_loop(self, node: ast.WhileStatement | ast.ForStatement):
        if node.test is None:
            return node
        
        test = self.visit(node.test)

        if not isinstance(test, ast.Constant):
            return node
        if not self._check_bool(test):
            return node
        
        if self._check_true(test):
            new = node.copy()
            new.test = None
            return new
        
        return None

    def visit_WhileStatement(self, node: ast.WhileStatement):
        return self._handle_loop(node)

    def visit_ForStatement(self, node: ast.ForStatement):
        return self._handle_loop(node)

    def visit_IfExpression(self, node: ast.IfExpression):
        test = self.visit(node.test)
        
        if not isinstance(test, ast.Constant):
            return node
        if not self._check_bool(test):
            return node

        if self._check_true(test):
            new = self.visit(node.body)
        else:
            new = self.visit(node.orelse)
        
        return self._apply_loc(node, new)

    def visit_UnaryExpression(self, node: ast.UnaryExpression):
        operand = self.visit(node.operand)
        
        if not isinstance(operand, ast.Constant):
            return node
        
        type, value = operand.type, operand.value

        is_int_float = isinstance(type, ast.IntType) or isinstance(type, ast.FloatType)
        is_bool = isinstance(type, ast.BoolType)

        match node.op:    
            case ast.PosOp() if is_int_float:
                new =  node.operand
            case ast.NegOp() if is_int_float:
                new = ast.Constant(value=-value, type=type)
            case ast.NotOp() if is_bool or is_int_float:
                new = ast.Constant(value=not value, type=ast.BoolType())
            case _:
                return node
        
        return self._apply_loc(node, new)
    
    def visit_BinaryExpression(self, node: ast.BinaryExpression):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)

        if not isinstance(lhs, ast.Constant) or not isinstance(rhs, ast.Constant):
            return node
        
        ltype, lvalue = lhs.type, lhs.value
        rtype, rvalue = rhs.type, rhs.value

        lhs_int = isinstance(ltype, ast.IntType)
        lhs_fixed = isinstance(ltype, ast.FixedType)
        lhs_float = isinstance(ltype, ast.FloatType)
        lhs_bool = isinstance(ltype, ast.BoolType)
        rhs_int = isinstance(rtype, ast.IntType)
        rhs_fixed = isinstance(rtype, ast.FixedType)
        rhs_float = isinstance(rtype, ast.FloatType)
        rhs_bool = isinstance(rtype, ast.BoolType)

        int_int = lhs_int and rhs_int
        float_float = lhs_float and rhs_float
        fixed_fixed = lhs_fixed and rhs_fixed
        int_float = (lhs_int and rhs_float) or (lhs_float and rhs_int)
        int_fixed = (lhs_int and rhs_fixed) or (lhs_fixed and rhs_int)
        float_fixed = (lhs_float and rhs_fixed) or (lhs_fixed and rhs_float)
        bool_bool = lhs_bool and rhs_bool
        comparable = int_int or float_float or int_float or fixed_fixed or int_fixed or float_fixed

        try:
            match node.op:
                case ast.AddOp() if int_int:
                    new = ast.Constant(value=lvalue + rvalue, type=ast.IntType())
                case ast.SubOp() if int_int:
                    new = ast.Constant(value=lvalue - rvalue, type=ast.IntType())
                case ast.MulOp() if int_int:
                    new = ast.Constant(value=lvalue * rvalue, type=ast.IntType())
                case ast.DivOp() if int_int:
                    new = ast.Constant(value=lvalue // rvalue, type=ast.IntType())
                case ast.ModOp() if int_int:
                    new = ast.Constant(value=lvalue % rvalue, type=ast.IntType())

                case ast.AddOp() if int_float or float_float or float_fixed:
                    new = ast.Constant(value=lvalue + rvalue, type=ast.FloatType())
                case ast.SubOp() if int_float or float_float or float_fixed:
                    new = ast.Constant(value=lvalue - rvalue, type=ast.FloatType())
                case ast.MulOp() if int_float or float_float or float_fixed:
                    new = ast.Constant(value=lvalue * rvalue, type=ast.FloatType())
                case ast.DivOp() if int_float or float_float or float_fixed:
                    new = ast.Constant(value=lvalue / rvalue, type=ast.FloatType())
                case ast.ModOp() if int_float or float_float or float_fixed:
                    new = ast.Constant(value=lvalue % rvalue, type=ast.FloatType())

                case ast.AddOp() if int_fixed or fixed_fixed:
                    new = ast.Constant(value=lvalue + rvalue, type=ast.FixedType())
                case ast.SubOp() if int_fixed or fixed_fixed:
                    new = ast.Constant(value=lvalue - rvalue, type=ast.FixedType())
                case ast.MulOp() if int_fixed or fixed_fixed:
                    new = ast.Constant(value=lvalue * rvalue, type=ast.FixedType())
                case ast.DivOp() if int_fixed or fixed_fixed:
                    new = ast.Constant(value=lvalue / rvalue, type=ast.FixedType())
                case ast.ModOp() if int_fixed or fixed_fixed:
                    new = ast.Constant(value=lvalue % rvalue, type=ast.FixedType())

                case ast.AndOp() if int_int or bool_bool:
                    new = ast.Constant(value=bool(lvalue and rvalue), type=ast.BoolType())
                case ast.OrOp() if int_int or bool_bool:
                    new = ast.Constant(value=bool(lvalue or rvalue), type=ast.BoolType())

                case ast.EqOp() if comparable:
                    new = ast.Constant(value=lvalue == rvalue, type=ast.BoolType())
                case ast.NeOp() if comparable:
                    new = ast.Constant(value=lvalue != rvalue, type=ast.BoolType())
                case ast.LtOp() if comparable:
                    new = ast.Constant(value=lvalue < rvalue, type=ast.BoolType())
                case ast.LeOp() if comparable:
                    new = ast.Constant(value=lvalue <= rvalue, type=ast.BoolType())
                case ast.GtOp() if comparable:
                    new = ast.Constant(value=lvalue > rvalue, type=ast.BoolType())
                case ast.GeOp() if comparable:
                    new = ast.Constant(value=lvalue >= rvalue, type=ast.BoolType())
                case _:
                    return node
                
        except ArithmeticError:
            return node
        
        return self._apply_loc(node, new)


if __name__ == "__main__":
    from vcs.cli import cli
    cli()
