import subprocess
import sys
import os
import fire
from typing import List

# def main(start: int = 32, end : int = 128 * 1024, bsz: int = 1, 
def main(start: int = 256, end : int = 128 * 1024, bsz: int = 1, 
        #  bench: List[str] = ['Baseline', 'CacheDecompressed', 'CacheCompressed']):
        #  bench: List[str] = ['B', 'CC', 'CD', 'A', 'A_CC', 'A_CC_ME', 'AM_CC_ME']):
         bench: List[str] = ['FI']):
    if isinstance(bench, str):
        bench = [bench]
    sys.stdout.write('bench,bsz,kv_len,device_name,cache_size,mean,median,p25,p75\n')
    sys.stdout.flush()
    for method in bench:
        cur = start
        while cur <= end:
            # print(f'Running {method} with kv_len={cur}', file=sys.stderr)
            ret = subprocess.call(['python3', '-m', 'mla.benchmark', method, str(cur), '--bsz', str(bsz) ,'--csv'], stdout=sys.stdout)
            if ret != 0:
                print(f'Failed with error: {ret}', file=sys.stderr)
                break
            cur *= 2

if __name__ == '__main__':
    fire.Fire(main)