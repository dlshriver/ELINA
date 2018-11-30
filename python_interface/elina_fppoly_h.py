"""
David Shriver
"""

from ctypes import *
from elina_auxiliary_imports import CtypesEnum


class ElinaLayerType(CtypesEnum):
    FFN = 0
    CONV = 1
    MAXPOOL = 2


class ElinaActivationType(CtypesEnum):
    RELU = 0
    SIGMOID = 1
    TANH = 2
    NONE = 3


class ElinaExprType(CtypesEnum):
    DENSE = 0
    SPARSE = 1


class ElinaExpr(Structure):
    _fields_ = [
        ("sup_coeff", POINTER(c_double)),
        ("inf_cst", c_double),
        ("sup_cst", c_double),
        ("type", c_uint8),  # TODO: Change to ElinaExprType?
        ("dim", POINTER(c_size_t)),
        ("size", c_size_t),
    ]


class ElinaNeuron(Structure):
    _fields_ = [
        ("lb", c_double),
        ("ub", c_double),
        ("expr", POINTER(ElinaExpr)),
        ("maxpool_lexpr", POINTER(ElinaExpr)),
        ("maxpool_uexpr", POINTER(ElinaExpr)),
    ]


class ElinaLayer(Structure):
    _fields_ = [
        ("dims", c_size_t),
        ("type", c_uint8),  # TODO: Change to ElinaLayerType?
        ("activation", c_uint8),  # TODO: Change to ElinaActivationType?
        ("neurons", POINTER(POINTER(ElinaNeuron))),
    ]


class ElinaOutputAbstract(Structure):
    _fields_ = [
        ("output_inf", POINTER(c_double)),
        ("output_sup", POINTER(c_double)),
        ("lexpr", POINTER(POINTER(ElinaExpr))),
        ("uexpr", POINTER(POINTER(ElinaExpr))),
    ]


class ElinaFPPoly(Structure):
    _fields_ = [
        ("layers", POINTER(POINTER(ElinaLayer))),
        ("num_layers", c_size_t),
        ("input_inf", POINTER(c_double)),
        ("input_sup", POINTER(c_double)),
        ("size", c_size_t),
        ("num_pixels", c_size_t),
        ("out", POINTER(ElinaOutputAbstract)),
    ]
