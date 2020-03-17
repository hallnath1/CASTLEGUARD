# k-anonymity
disable-dp:
	python3 src/main.py --filename example.csv --sample-size 200 --delta 40 --k 10 --beta 10 --l 1 --disable-dp --display --seed 42

# l-diversity
ldiverse:
	python3 src/main.py --filename example.csv --sample-size 200 --delta 40 --k 10 --beta 10 --l 5 --disable-dp --display --seed 42

# Differential privacy
enable-dp:
	python3 src/main.py --filename example.csv --sample-size 200 --delta 40 --k 10 --beta 10 --l 5 --phi 100 --display --seed 42

# Issues with a small phi
small-phi:
	python3 src/main.py --filename example.csv --sample-size 200 --delta 40 --k 10 --beta 10 --l 5 --phi 1 --display --seed 42

# Nice graph
demo:
	python3 src/main.py --filename example.csv --sample-size 1000 --k 10 --l 5 --delta 200 --seed 42 --display
