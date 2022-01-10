get_deps:
	pip3 install -r requirements.txt

clean:
	cd intent/explainers/parsers/; rm -rf *.so *.c *.html build/ __pycache__; cd -

build:
	cd intent/explainers/parsers/; python3 setup.py build_ext --inplace; cd ..

all: clean get_deps build
