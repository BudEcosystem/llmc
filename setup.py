from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename) as f:
        requirements = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
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
