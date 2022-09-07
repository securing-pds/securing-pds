clean:
	rm -rf plots/*.png
	rm -rf plots/*.tex
	rm -rf plots/*.tikz

original:
	sage load_factor.py --trials 16 --type original --out load_factor_original.json --ncpus 16 --no-run

prfwrapped:
	sage load_factor.py --trials 16 --type prfwrapped --out load_factor_prfwrapped.json --ncpus 16 --no-run

secinstcuckoo:
	sage correctness.py --cuckoo

secinstbloom:
	sage correctness.py --bloom

