from joblib import Parallel, delayed

import numpy as np


def func1(a, b):

    result = Parallel(n_jobs=1, verbose=0)(
        delayed(func2)(a_elem, b_elem) for a_elem, b_elem in zip(a, b)
    )
    print(result)


def func2(a, b):

    c = np.arange(a)
    d = np.arange(b)

    #print('c: ', c)
    #print('d: ', d)

    result = Parallel(n_jobs=1, verbose=0)(
        delayed(func3)(c_elem, d_elem) for c_elem, d_elem in zip(c, d)
    )

    return result


def func3(c, d):

    return c * d


if __name__ == '__main__':

    a = np.arange(4)
    b = np.arange(4)

    score = 0
    for a_elem in a:
        for b_elem in b:

            score += a_elem * b_elem


    func1(a, b)
