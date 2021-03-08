#!/usr/bin/env python


# Return LaTeX name for a line / complex name
def texlinename(name) -> str:
    if name == 'Ha':
        tname = r'H$\alpha$'
    elif name == 'Hb':
        tname = r'H$\beta$'
    elif name == 'Hr':
        tname = r'H$\gamma$'
    elif name == 'Hg':
        tname = r'H$\gamma$'
    elif name == 'Lya':
        tname = r'Ly$\alpha$'
    else:
        tname = name
    return tname