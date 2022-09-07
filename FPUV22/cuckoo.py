from sage.all import binomial, parallel
from collections import OrderedDict
import multiprocessing

class parallel_params:
    NCPUS = multiprocessing.cpu_count() - 1
    TRIALS = 2**12
    MAX_FILES = 100


class bcolors:
    Black = '\u001b[30m'
    Red = '\u001b[31m'
    Green = '\u001b[32m'
    Yellow = '\u001b[33m'
    Blue = '\u001b[34m'
    Magenta = '\u001b[35m'
    Cyan = '\u001b[36m'
    White = '\u001b[37m'

def empty_matrix(rows, cols, bot=None):
    m = []
    for r in range(rows):
        m.append([bot] * cols)
    return m

def empty_3Dmatrix(rows, cols, depth, bot=None):
    m = []
    for r in range(rows):
        m.append([[bot] * depth] * cols)
    return m

def empty_4Dmatrix(rows, cols, depth, width, bot=None):
    m = []
    for r in range(rows):
        m.append([[[bot] * width] * depth] * cols)
    return m

def xor_bytes(b1, b2):
    return bytes(a ^ b for a, b in zip(b1, b2))


def popcount(x):
    return bin(x).count('1')


def x_is_mul_of_y(x, y):
    return bool((x//y)*y == x)


def slice_list(input_l, size):
    # https://stackoverflow.com/a/4119142
    input_size = len(input_l)
    slice_size = input_size // size
    remain = input_size % size
    result = []
    iterator = iter(input_l)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(next(iterator))
        if remain:
            result[i].append(next(iterator))
            remain -= 1
    return result


import hashlib
class Hash:

    def __init__(self, hashlen, salt=None):
        self.hashlen = hashlen
        self.salt = salt
        if not isinstance(salt, bytes) and salt != None:
            raise ValueError('Hash.salt should be a `bytes` buffer or `None`.')

    def eval(self, x):
        if not isinstance(x, bytes):
            raise ValueError('Argument to eval should be a `bytes` buffer.')
        h = hashlib.shake_128()
        if self.salt:
            h.update(self.salt)
        h.update(x)
        return h.digest(self.hashlen)


class PRNG:

    def __init__(self, seed):
        assert(isinstance(seed, int))
        self.seed = seed
        self.iv = Hash(64).eval(self.seed.to_bytes(32, 'big'))
        self.counter = 0

    def sample(self, bytes, returntype='bytes'):
        cnt = self.counter.to_bytes(32, 'big')
        self.counter += 1
        h = hashlib.shake_128()
        h.update(self.iv)
        h.update(cnt)
        if returntype == 'bytes':
            return h.digest(bytes)
        elif returntype == 'int':
            return int.from_bytes(h.digest(bytes), 'big')
        else:
            raise ValueError('returntype should be bytes or int')


class CuckooFilter:

    def __init__(self, m_len, b_len, kickmax, tag_len, prng):
        """ 
            Current restrictions:
            m = bytes
            taglen = bytes
        """
        self.m_len = m_len
        self.m_bytes = (self.m_len-1)//8+1
        self.m = 2**self.m_len
        self.b_len = b_len
        self.b_bytes = (self.b_len-1)//8+1
        self.b = 2**self.b_len
        self.kickmax = kickmax
        self.tag_len = tag_len
        self.tag_bytes = (self.tag_len-1)//8+1
        self.tag_space = 2**self.tag_len
        self.M = empty_matrix(self.m, self.b)
        self._H_I = Hash(self.m_bytes, salt=b'H_I'+prng.sample(8))
        self._H_T = Hash(self.tag_bytes, salt=b'H_T'+prng.sample(8))
        self.prng = prng
        self.evicted = None

    def H_I(self, x):
        hi = int.from_bytes(self._H_I.eval(bytes(x, encoding='utf8')), 'big') 
        hi = hi & (self.m-1)
        ret = hi.to_bytes(self.m_bytes, 'big')
        return ret

    def H_T(self, x):
        ht = int.from_bytes(self._H_T.eval(bytes(x, encoding='utf8')), 'big')
        ht = ht & (self.tag_space-1)
        ret = ht.to_bytes(self.tag_bytes, 'big')
        return ret

    def _get_other_index(self, tag, i1):
        _i1 = i1
        if not isinstance(i1, bytes):
            _i1 = i1.to_bytes(self.m_bytes, 'big')
        i2 = xor_bytes(_i1, self.H_I(str(tag)))
        return i2

    def _get_indices(self, x):
        tag = self.H_T(x)
        i1 = self.H_I(x)
        i2 = self._get_other_index(tag, i1)
        return tag, (int.from_bytes(i1, 'big'), int.from_bytes(i2, 'big'))

    def insert(self, x):
        x = str(x)
        tag, i = self._get_indices(x)
        if self.evicted:
            return False
        # if already in, return true
        if self.check(x):
            return True
        # if free room in one of the bins, store and return true
        for b in [0, 1]:
            if None in self.M[i[b]]:
                idx = self.M[i[b]].index(None)
                self.M[i[b]][idx] = tag
                return True
        # else kick something from a random bin
        b = self.prng.sample(1, returntype='int') & 1
        j = i[b]
        for _ in range(self.kickmax):
            slot = self.prng.sample(self.b_bytes, returntype='int') & (2**self.b_len-1)
            elem = self.M[j][slot]
            self.M[j][slot] = tag
            tag = elem
            j = int.from_bytes(self._get_other_index(tag, j), 'big')
            if None in self.M[j]:
                idx = self.M[j].index(None)
                self.M[j][idx] = tag
                return True
        self.evicted = tag
        return True

    def check(self, x, verbose=False):
        x = str(x)
        tag, i = self._get_indices(x)
        if verbose:
            print (bin(int.from_bytes(tag, 'big')), i[0], i[1])
        found = (tag in self.M[i[0]]) or (tag in self.M[i[1]]) or (tag == self.evicted)
        return found

    def reveal(self):
        fmt_str = "{:<4}  " + ("{:<%d}  "%(self.tag_len+2) * (self.b))
        for i in range(self.m):
            print(fmt_str.format(i, *["⟂" if x == None else bin(int.from_bytes(x, 'big')) for x in self.M[i]]))
        return self.M
    
    def up_disabled(self):
        return self.evicted != None


class StreamlinedCF(CuckooFilter):

    def __init__(self, m_len, b_len, kickmax, tag_len, prng):
        CuckooFilter.__init__(self, m_len, b_len, kickmax, tag_len, prng)
        self._F = Hash(self.m_bytes + self.tag_bytes, salt=b'F'+prng.sample(8))
        self._G = Hash(self.m_bytes, salt=b'G'+prng.sample(8))
        self.F_range_space = self.m * self.tag_space

    def F(self, x):
        hi = int.from_bytes(self._F.eval(bytes(x, encoding='utf8')), 'big')
        hi = hi & (self.F_range_space-1)
        ret = hi.to_bytes(self.m_bytes + self.tag_bytes, 'big')
        return ret

    def G(self, x):
        ht = int.from_bytes(self._G.eval(bytes(x, encoding='utf8')), 'big')
        ht = ht & (self.m-1)
        ret = ht.to_bytes(self.m_bytes, 'big')
        return ret

    def _get_other_index(self, tag, i1):
        _i1 = i1
        if not isinstance(i1, bytes):
            _i1 = i1.to_bytes(self.m_bytes, 'big')
        i2 = xor_bytes(_i1, self.G(str(tag)))
        return i2

    def _get_indices(self, x):
        i_tag_i1 = int.from_bytes(self.F(x), 'big')
        i_i1 = i_tag_i1 & (self.m-1)
        i_tag = (i_tag_i1 >> self.m_len) & (self.tag_space-1)
        i1 = i_i1.to_bytes(self.m_bytes, 'big')
        tag = i_tag.to_bytes(self.tag_bytes, 'big')

        # divide into top and bottom bits
        i2 = self._get_other_index(tag, i1)
        return tag, (int.from_bytes(i1, 'big'), int.from_bytes(i2, 'big'))


class PRFWrappedCF(CuckooFilter):

    def __init__(self, m_len, b_len, kickmax, tag_len, prng):
        CuckooFilter.__init__(self, m_len, b_len, kickmax, tag_len, prng)
        self.F_len = 256
        self.F_bytes = self.F_len // 8
        self._F = Hash(self.F_bytes, salt=b'F'+prng.sample(8))

    def F(self, x):
        ret = self._F.eval(bytes(x, encoding='utf8'))
        return ret

    def _get_indices(self, x):
        y = str(self.F(x))
        return CuckooFilter._get_indices(self, y)


instances = [
    # # {'m_len': 4, 'b_len': 2, 'kickmax': 5, 'tag_len': 4}
    {'m_len': 4, 'b_len': 1, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 4, 'b_len': 2, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 4, 'b_len': 3, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 5, 'b_len': 1, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 5, 'b_len': 2, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 5, 'b_len': 3, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 6, 'b_len': 1, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 6, 'b_len': 2, 'kickmax': 5, 'tag_len': 8},
    {'m_len': 6, 'b_len': 3, 'kickmax': 5, 'tag_len': 8},
    # {'m_len': 4, 'b_len': 3, 'kickmax': 20, 'tag_len': 8},
    # {'m_len': 4, 'b_len': 3, 'kickmax': 50, 'tag_len': 8},
    # {'m_len': 4, 'b_len': 3, 'kickmax': 100, 'tag_len': 8},
    # {'m_len': 4, 'b_len': 3, 'kickmax': 250, 'tag_len': 8},
    # {'m_len': 4, 'b_len': 3, 'kickmax': 500, 'tag_len': 8},
]


def main():
    prng = PRNG(0xdeadbeef)
    for params in instances:
        m_len, b_len, kickmax, tag_len = params['m_len'], params['b_len'], params['kickmax'], params['tag_len']
        # cf = CuckooFilter(m_len, b_len, kickmax, tag_len, prng)
        # cf = StreamlinedCF(m_len, b_len, kickmax, tag_len, prng)
        cf = PRFWrappedCF(m_len, b_len, kickmax, tag_len, prng)
        cf.reveal()
        for char in map(chr, range(ord('a'), ord('z')+1)):
            check_before = cf.check(char)
            cf.insert(char)
            check_after = cf.check(char)
            cf.reveal()
            print(f'{char}    before: {check_before}, after: {check_after}.   Press enter to continue.')
            input()


def fpr_n_theory(params, n_list):
    m_len, b_len, kickmax, tag_len = params['m_len'], params['b_len'], params['kickmax'], params['tag_len']
    b = 2**b_len
    m = 2**m_len

    def inner(tag_len, m, n, w, target_bins):
        return (1-(1-1/2**tag_len)**w) * (target_bins/m)**w * (1-target_bins/m)**(n-w) * binomial(n, w)

    d = OrderedDict()
    for n in n_list:
        d[n] = (1-2**(-m_len)) * sum(inner(tag_len, m, n, w, 2) for w in range(2*b+1)) + 2**(-m_len) * sum(inner(tag_len, m, n, w, 1) for w in range(b+1))
    d['evic'] = -1

    return d


def single_thread_experiment_batch(prng_seed_batch, m_len, b_len, kickmax, tag_len, n_list):
    d = OrderedDict()
    tr = OrderedDict()
    for n in n_list:
        d[n] = tr[n] = 0
    d['evic'] = 0

    for prng_seed in prng_seed_batch:
        prng = PRNG(prng_seed)
        cf = CuckooFilter(m_len, b_len, kickmax, tag_len, prng)
        up_disabled = False
        for x in range(max(n_list)+1):
            if x > 0:
                ins = cf.insert(x)
                if not ins and not up_disabled:
                    up_disabled = x-1
                    break
            if x in n_list:
                tr[x] += 1
                qry = cf.check("definitely this is not in", verbose=False)
                if qry:
                    d[x] += 1
        d['evic'] += up_disabled
    return (d, tr)


@parallel(ncpus=parallel_params.NCPUS)
def parallelExperiment(experiment_params):
    """ Run in parallel a batch of single thread experiments.
        This is just a wrapper function, and exists because of the @parallel decorator
        handling easily only functions that take a single input.
    """
    prng_seed_batch, m_len, b_len, kickmax, tag_len, n_list = experiment_params
    _d, _tr = single_thread_experiment_batch(prng_seed_batch, m_len, b_len, kickmax, tag_len, n_list)

    return _d, _tr


def fpr_n_experiment(params, n_list, trials=2**12, batches=parallel_params.NCPUS):
    m_len, b_len, kickmax, tag_len = params['m_len'], params['b_len'], params['kickmax'], params['tag_len']

    d = OrderedDict()
    tr = OrderedDict()
    for n in n_list:
        d[n] = 0
        tr[n] = 0
    d['evic'] = 0

    prng_seeds = list(range(1,trials+1))
    prng_seed_batches = slice_list(prng_seeds, batches)

    ret = list(parallelExperiment(([prng_seed_batch, m_len, b_len, kickmax, tag_len, n_list] for prng_seed_batch in prng_seed_batches)))

    for cpu in range(batches):
        _d, _tr = ret[cpu][1]
        for key in _d: d[key] += _d[key]
        for key in _tr: tr[key] += _tr[key]

    for n in d:
        if n != 'evic' and tr[n] != 0:
            d[n] /= tr[n]

    return d


if __name__ == "__main__":
    main()
