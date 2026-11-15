"""Read Strata's expert routing profile (STRP v1) and print coverage numbers.
Format (strata include/strata/core/expert_cache.hpp): 'STRP', u32 version, n_layers, n_expert, slots, n_ranked,
n_ranked x (u16 layer, u16 expert) ranked by descending routing frequency, then n_layers x n_expert int32 table
(tools/make_profile.py: frequency counts; --expert-profile-save: rank, -1 = unranked).
usage: python -m tt_qwen38.expert_profile tt_qwen38/data/expert-profile-coder.prof"""
import struct, sys
import numpy as np

def read(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'STRP', 'not a STRP profile'
    ver, nl, ne, slots, nr = struct.unpack_from('<5I', b, 4)
    assert ver == 1
    pairs = np.frombuffer(b, '<u2', nr * 2, 24).reshape(nr, 2).astype(int)
    table = np.frombuffer(b, '<i4', nl * ne, 24 + nr * 4).reshape(nl, ne)
    return dict(n_layers=nl, n_expert=ne, slots=slots, ranked=pairs, table=table)

if __name__ == '__main__':
    p = read(sys.argv[1]); t = p['table']; r = p['ranked']
    print(f"layers {p['n_layers']} experts {p['n_expert']} slots {p['slots']} ranked {len(r)}")
    n = t.size
    is_rank = t.min() == 0 and t.max() == n - 1 and int(t.astype(np.int64).sum()) == n * (n - 1) // 2
    print('table is a RANK table (0 = hottest), not counts:' if is_rank else 'table looks like counts:', is_rank)
    if is_rank:
        assert all(t[l, e] == i for i, (l, e) in enumerate(r)), 'ranked list disagrees with rank table'
        print('ranked list matches the rank table; no routing frequencies in this file')
    per_exp = 3 * 2560 * 640 * 0.5625  # BFP4 bytes per expert (gate+up+down) = 2,764,800
    for gb in (12, 16, 20, 22):
        k = min(int(gb * 1e9 // per_exp), len(r)); top = r[:k]
        cnt = np.bincount(top[:, 0], minlength=p['n_layers'])
        print(f'  top {k:5d} pairs ({gb} GB BFP4, {k / n:.1%}): per-layer count min {cnt.min()} median {int(np.median(cnt))} max {cnt.max()}')
