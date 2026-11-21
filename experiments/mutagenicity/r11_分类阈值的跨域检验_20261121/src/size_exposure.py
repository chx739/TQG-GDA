"""One-parameter Bernoulli size curve and registered R11 decision arms."""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as F
from math import fsum

import numpy as np
from scipy.optimize import brentq

from . import roc_rate, thresholds

BASE = ('B0','Size','OldPool','OldDirectP')
CONTROLS = ('PoolOther1','PoolOther2','PriorOther1','PriorOther2','QuantileOther1','QuantileOther2')
ARMS = (*BASE,'Curve','CurvePool','CurveDirectP','SourceMajority',*CONTROLS)
MEANS = {f'{m}OtherMean':(f'{m}Other1',f'{m}Other2') for m in ('Pool','Prior','Quantile')}


def fit(nodes, labels):
    n,y = np.asarray(nodes),np.asarray(labels)
    if (n.ndim!=1 or y.shape!=n.shape or not len(n) or not np.issubdtype(n.dtype,np.integer)
            or np.any(n<=0) or not np.isin(y,[0,1]).all()):
        raise ValueError('Fit requires positive integer node counts and binary S labels')
    pos,neg = int(y.sum()),len(y)-int(y.sum())
    result = dict(n_source=len(y),positive=pos,negative=neg,min_nodes=int(n.min()),max_nodes=int(n.max()))
    if not pos or not neg:
        return dict(**result,status='constant_class',rate=0. if pos else None,constant=int(bool(pos)),
                    converged=True,iterations=0,nll=0.,derivative=None)
    exposure = sum(map(int,n[y==1]))
    neg_n = n[y==0].astype(np.float64)
    sizes,counts = np.unique(n[y==0],return_counts=True)
    negatives = [[int(a),int(b)] for a,b in zip(sizes,counts,strict=True)]
    upper = neg/exposure

    def gradient(rate):
        if not rate:
            return -np.inf
        with np.errstate(over='ignore',divide='ignore'):
            return exposure-float(np.sum(neg_n/np.expm1(rate*neg_n)))

    if n.min()==n.max():
        rate = -np.log(pos/len(y))/int(n[0])
        status,iterations = 'equal_node_count',0
    else:
        rate,solution = brentq(gradient,0.,upper,xtol=np.nextafter(0.,1.),full_output=True)
        if not solution.converged:
            raise ArithmeticError('Size exposure likelihood failed to converge')
        status,iterations = 'brentq',int(solution.iterations)
    error = gradient(rate)
    if not np.isfinite(error) or abs(error)>1e-10*exposure:
        raise ArithmeticError('Size curve failed the source score equation')
    value = rate*exposure-float(np.log(-np.expm1(-rate*neg_n)).sum())
    return dict(**result,status=status,rate=float(rate),constant=None,converged=True,iterations=iterations,
                derivative=error,nll=value/len(y),positive_exposure=exposure,negative_sizes=negatives)


def threshold_sign(q,model):
    """Sign of b(q)-1/2 from the monotone likelihood derivative at log(2)/q."""
    negatives,exposure = model['negative_sizes'],model['positive_exposure']
    if all(n%q==0 for n,_ in negatives):
        s = sum(F(n*k,(1 << (n//q))-1) for n,k in negatives)
        return int(exposure>s)-int(exposure<s)
    # With t=2**(1/q), geometric expansion of 1/(t**n-1)
    # modulo t**q=2 has nonnegative coefficients and a nonzero
    # nonconstant coefficient if q does not divide n. Irreducibility
    # of t**q-2 means the sum cannot then equal the integer exposure.
    # Bound the sum with directed rounding until it separates from exposure.
    precision = 48
    while True:
        with localcontext() as ctx:
            ctx.prec = precision
            log2 = Decimal(2).ln()

            def bound(upper):
                total = Decimal(0)
                ln_bound = log2.next_minus() if upper else log2.next_plus()
                for n,k in negatives:
                    ctx.rounding = ROUND_FLOOR if upper else ROUND_CEILING
                    exp = (ln_bound*n/q).exp()
                    den = (exp.next_minus() if upper else exp.next_plus())-1
                    ctx.rounding = ROUND_CEILING if upper else ROUND_FLOOR
                    total += Decimal(n*k)/den
                return total

            if exposure>bound(True):
                return 1
            if exposure<bound(False):
                return -1
        precision *= 2


def predict(nodes, model):
    n = np.asarray(nodes)
    if n.ndim!=1 or not np.issubdtype(n.dtype,np.integer) or np.any(n<=0):
        raise ValueError('Predict requires positive integer node counts')
    c = model['constant']
    if c is not None:
        return np.full(len(n),c,dtype=np.float64)
    p = np.exp(-model['rate']*n)
    if model['status']=='equal_node_count':
        # At the observed exposure the MLE is the exact empirical prior.
        p[n==model['min_nodes']] = float(F(model['positive'],model['n_source']))
    for q in np.unique(n[np.abs(p-.5)<=32*np.spacing(.5)]):
        sign = threshold_sign(int(q),model)
        clip = n==q
        if not sign:
            p[clip] = .5
        elif sign==1 and np.any(p[clip]<=.5):
            p[clip] = np.nextafter(.5,1.)
        elif sign==-1 and np.any(p[clip]>=.5):
            p[clip] = np.nextafter(.5,0.)
    return p


def pool_probability(p):
    if not len(p) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Need a nonempty vector of probabilities for the original A subset')
    return F.from_float(float(np.mean(p)))


def mixed_pool_sign(sizes,counts,model):
    """Certify the sign at the fitted exposure root if a mixed A mean is near 1/2."""
    from sympy import Poly, Symbol, ZZ

    x = Symbol('x')
    n_a = sum(counts)
    target = Poly(2*sum(c*x**n for n,c in zip(sizes,counts,strict=True))-n_a,x,domain=ZZ)
    negatives,exposure = model['negative_sizes'],model['positive_exposure']
    # x=exp(-lambda). Clearing the score denominator introduces no root in (0,1).
    numerator,denominator = Poly(exposure,x,domain=ZZ),Poly(1,x,domain=ZZ)
    for n,c in negatives:
        d = Poly(1-x**n,x,domain=ZZ)
        numerator = (numerator*d-denominator*Poly(n*c*x**n,x,domain=ZZ)).rem(target)
        denominator = (denominator*d).rem(target)
    if target.gcd(numerator).count_roots(0,1):
        return 0

    def compare_at(p):
        value = 2*sum(c*p**n for n,c in zip(sizes,counts,strict=True))-n_a
        return int(value>0)-int(value<0)

    # Both equations have unique roots. Bisect the fit root using the exact
    # rational score and stop as soon as the interval is separated from the pool root.
    left,right = F(0),F(1)
    while True:
        m = (left+right)/2
        z = [m**n for n,_ in negatives]
        score = exposure-sum(n*c*p/(1-p) for (n,c),p in zip(negatives,z,strict=True))
        if not score:
            return compare_at(m)
        if score>0:
            if compare_at(m)>=0:
                return 1
            left = m
        else:
            if compare_at(m)<=0:
                return -1
            right = m


def exposure_prior(nodes,model):
    sizes,counts = np.unique(nodes,return_counts=True)
    probabilities = predict(sizes,model)
    ns,cs = list(map(int,sizes)),list(map(int,counts))
    if not cs:
        raise ValueError('The authenticated A pool is empty')
    origin = model['min_nodes']
    if model['status']=='equal_node_count' and all(n%origin==0 for n in ns):
        p = F(model['positive'],model['n_source'])
        return sum(c*p**(n//origin) for n,c in zip(ns,cs,strict=True))/sum(cs)
    p = fsum(float(x)*c for x,c in zip(probabilities,cs,strict=True))/sum(cs)
    if model['constant'] is None and abs(p-.5)<=32*np.spacing(.5):
        sign = threshold_sign(ns[0],model) if len(ns)==1 else mixed_pool_sign(ns,cs,model)
        if not sign:
            return F(1,2)
        if (p>.5)-(p<.5)!=sign:
            p = np.nextafter(.5,1. if sign>0 else 0.)
    return F.from_float(float(p))


def policies(target, p_by_target, a_by_target, held, old_p, p_h, model):
    own_p = p_by_target[target]
    choices,details = {},{}

    def pool(name,p):
        choices[name] = np.full(len(held),p>F(1,2),dtype=bool)
        details[name] = dict(p=str(p))

    def quantile(name,p,ref):
        cut,n_positive = roc_rate.quantile(a_by_target[ref],p)
        choices[name] = roc_rate.classify(held,cut)
        details[name] = dict(p=str(p),reference=ref,cut=thresholds.encode_threshold(cut),
                             a_positive=n_positive)

    choices['Curve'] = p_h>.5
    pool('SourceMajority',F(model['positive'],model['n_source']))
    pool('OldPool',old_p)
    pool('CurvePool',own_p)
    quantile('OldDirectP',old_p,target)
    quantile('CurveDirectP',own_p,target)
    others = sorted(set(a_by_target)-{target})
    if len(others)!=2 or set(a_by_target)!=set(p_by_target):
        raise ValueError('Expected exactly three same-source reference pools')
    for i,foreign in enumerate(others,1):
        p = p_by_target[foreign]
        pool(f'PoolOther{i}',p)
        details[f'PoolOther{i}']['reference'] = foreign
        quantile(f'PriorOther{i}',p,target)
        details[f'PriorOther{i}']['prior_donor'] = foreign
        quantile(f'QuantileOther{i}',own_p,foreign)
    return choices,details
