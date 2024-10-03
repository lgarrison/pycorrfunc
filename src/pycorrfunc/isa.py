def _lookup_isa(isa, _pycorrfunc):
    # isa_t is defined in both modules,
    # so just use the one that's already imported

    isa_t = _pycorrfunc.isa_t
    isa = isa.upper()
    try:
        ret = isa_t[isa]
    except KeyError as e:
        raise ValueError(
            f'ISA "{isa}" not recognized. Must be one of {[v.name for v in isa_t]}'
        ) from e
    return ret
