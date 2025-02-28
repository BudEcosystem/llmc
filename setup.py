from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue
            # Skip options like '-r requirements.txt'
            if line.startswith('-'):
                continue
            requirements.append(line)
    return requirements

setup(
    name='llmc',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements/runtime.txt'),
)

