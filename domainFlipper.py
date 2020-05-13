import nifty5 as ift

class DomainFlipper(ift.LinearOperator):
    """
    Operator that changes a field's domain to its default codomain
    """
    def __init__(self, domain, target=None):
        self._domain = ift.DomainTuple.make(domain)
        if target is None:
            self._target = ift.DomainTuple.make(domain.get_default_codomain())
        else:
            self._target = ift.DomainTuple.make(target)
        self._capability = self._all_ops
        return

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            y = ift.from_global_data(self._target, x.to_global_data())
        if mode == self.INVERSE_TIMES:
            y = ift.from_global_data(self._domain, x.to_global_data())
        if mode == self.ADJOINT_TIMES:
            y = ift.from_global_data(self._domain, x.to_global_data())
        if mode == self.ADJOINT_INVERSE_TIMES:
            y = ift.from_global_data(self._target, x.to_global_data())
        return y
