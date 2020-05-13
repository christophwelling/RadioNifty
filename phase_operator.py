import numpy as np
import nifty5 as ift

class LinearSlopeOperator(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(ift.UnstructuredDomain((2,)))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        pos = np.zeros_like(self.target[0].get_k_length_array().val)
        #pos[:len(pos)//2+1] = self.target[0].get_k_length_array().val[:len(pos)//2+1]
        #pos[len(pos)//2+1:] = -1.*self.target[0].get_k_length_array().val[len(pos)//2+1:]
        pos = self.target[0].get_k_length_array().val

        self._pos = pos

    def apply(self, x, mode):
        self._check_input(x, mode)
        inp = x.to_global_data()
        if mode == self.TIMES:
            res = np.empty(self.target.shape, dtype=x.dtype)
            #res[0] = 0
            res = inp[1] + inp[0]*self._pos
        else:
            res = np.array(
                [np.sum(self._pos*inp),
                 np.sum(inp[1:])], dtype=x.dtype)
        return ift.Field.from_global_data(self._tgt(mode), res)

def SlopeSpectrumOperator(target, m=0, n=0, sigma_m=.1, sigma_n=.1):
    codomain = target.get_default_codomain()

    pos_diagonals = np.ones(target.shape[0])
    pos_diagonals[target.shape[0]//2+1:] = -1
    #pos_diagonals[0] = 0
    flipper = ift.DiagonalOperator(ift.Field(ift.DomainTuple.make(codomain), pos_diagonals))
    slope = LinearSlopeOperator(target.get_default_codomain())
    mean = np.array([m, n])
    sig = np.array([sigma_m, sigma_n])
    mean = ift.Field.from_global_data(slope.domain, mean)
    sig = ift.Field.from_global_data(slope.domain, sig)
    linear_operator = flipper @ slope @ ift.Adder(mean) @ ift.makeOp(sig)
    return linear_operator.ducktape('slope')
