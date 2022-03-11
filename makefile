get_deps:
	pip3 install -r requirements.txt

clean:
	cd ibug/parsers/; rm -rf *.so *.c *.html build/ __pycache__; cd -

build:
	cd ibug/parsers/; python3 setup.py build_ext --inplace; cd ..

package:
	rm -rf dist/
	python3 setup.py sdist bdist_wheel
	twine check dist/*

upload_pypi_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_pypi:
	twine upload dist/*

all: clean get_deps build
