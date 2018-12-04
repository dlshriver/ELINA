"""
David Shriver
"""

from ctypes import *
from elina_abstract0_h import ElinaAbstract0, ElinaAbstract0Ptr
from elina_auxiliary_imports import CtypesEnum


class LayerType(CtypesEnum):
    FFN = 0
    CONV = 1
    MAXPOOL = 2


class ActivationType(CtypesEnum):
    RELU = 0
    SIGMOID = 1
    TANH = 2
    NONE = 3


class ExprType(CtypesEnum):
    DENSE = 0
    SPARSE = 1


class Expr(Structure):
    _fields_ = [
        ("inf_coeff", POINTER(c_double)),
        ("sup_coeff", POINTER(c_double)),
        ("inf_cst", c_double),
        ("sup_cst", c_double),
        ("type", c_uint),
        ("dim", POINTER(c_size_t)),
        ("size", c_size_t),
    ]


class Neuron(Structure):
    _fields_ = [
        ("lb", c_double),
        ("ub", c_double),
        ("expr", POINTER(Expr)),
        ("maxpool_lexpr", POINTER(Expr)),
        ("maxpool_uexpr", POINTER(Expr)),
    ]


class Layer(Structure):
    _fields_ = [
        ("dims", c_size_t),
        ("type", c_uint),  # TODO: Change to ElinaLayerType?
        ("activation", c_uint),  # TODO: Change to ElinaActivationType?
        ("neurons", POINTER(POINTER(Neuron))),
    ]


class OutputAbstract(Structure):
    _fields_ = [
        ("output_inf", POINTER(c_double)),
        ("output_sup", POINTER(c_double)),
        ("lexpr", POINTER(POINTER(Expr))),
        ("uexpr", POINTER(POINTER(Expr))),
    ]


class FPPoly(Structure):
    _fields_ = [
        ("layers", POINTER(POINTER(Layer))),
        ("num_layers", c_size_t),
        ("input_inf", POINTER(c_double)),
        ("input_sup", POINTER(c_double)),
        ("size", c_size_t),
        ("num_pixels", c_size_t),
        ("out", POINTER(OutputAbstract)),
    ]


FPPolyPtr = POINTER(FPPoly)


def fppoly_of_abstract0(abs0: ElinaAbstract0Ptr) -> FPPoly:
    return cast(abs0[0].value, FPPolyPtr)


def expr_print(expr: Expr):
    print("[%g, %g]" % (-expr.inf_cst, expr.sup_cst), end="")
    if not expr.inf_coeff or not expr.sup_coeff:
        print()
        return
    total_eq_inf_sup = 0
    for i in range(expr.size):
        print(
            " + [%g, %g]x%d"
            % (
                -expr.inf_coeff[i],
                expr.sup_coeff[i],
                i if expr.type == ExprType.DENSE else expr.dim[i],
            ),
            end="",
        )
        total_eq_inf_sup += (-expr.inf_coeff[i] == expr.sup_coeff[i])
    print()
    print(total_eq_inf_sup)

