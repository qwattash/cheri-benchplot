from setuptools import setup, find_packages

setup(
    name="pycheribenchplot",
    version="1.0",
    packages=find_packages(),
    scripts=["benchplot.py"],
    install_requires=[
        "asyncssh>=2.7.2",
        "dataclasses-json>=0.5.6",
        "isort>=5.10.0",
        "Jinja2>=3.0.2",
        "matplotlib>=3.4.3",
        "numpy>=1.21.3",
        "openpyxl>=3.0.9",
        "pandas>=1.3.4",
        "pyelftools>=0.27",
        "PyPika>=0.48.8",
        "sortedcontainers>=2.4.0",
        "termcolor>=1.1.0",
        "XlsxWriter>=3.0.2",
        "yapf>=0.31.0",
        "gitpython>=3.1.27",
    ]
)
