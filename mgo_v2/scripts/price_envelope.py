"""Exact lower envelope of affine resource prices; rational arithmetic only."""
from fractions import Fraction


def fraction_record(x):
    return dict(numerator=x.numerator,denominator=x.denominator,decimal=float(x))


def lower_envelope(lines):
    # lines = (label, intercept, slope). Do not assume adjacent policies survive.
    lines=[(label,Fraction(a),Fraction(b)) for label,a,b in lines]
    assert len({label for label,_,_ in lines})==len(lines)
    roots={Fraction(0)}
    for _,a,b in lines:
        for _,c,d in lines:
            if b!=d:
                x=(c-a)/(b-d)
                if x>0:roots.add(x)
    roots=sorted(roots)
    regions=[]
    def winners(x):
        costs=[(label,a+b*x) for label,a,b in lines];best=min(v for _,v in costs)
        return [label for label,v in costs if v==best]
    for i,left in enumerate(roots):
        right=roots[i+1] if i+1<len(roots) else None
        selected=winners((left+right)/2 if right is not None else left+1)
        if regions and regions[-1]['winners']==selected:regions[-1]['end']=right
        else:regions.append(dict(start=left,end=right,winners=selected))
    crossings=[dict(price=r['start'],winners=winners(r['start'])) for r in regions[1:]]
    return regions,crossings


def serial_envelope(lines):
    regions,crossings=lower_envelope(lines)
    return dict(regions=[dict(start=fraction_record(r['start']),end=fraction_record(r['end']) if r['end'] is not None else None,winners=r['winners']) for r in regions],crossovers=[dict(price=fraction_record(r['price']),winners=r['winners']) for r in crossings])
