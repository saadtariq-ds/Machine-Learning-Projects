from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file=file_path, mode='r') as file:
        requirements = file.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name='student_performance_indicator',
    version='0.0.1',
    author='Saad',
    author_email='tariqsaad1997@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(file_path='requirements.txt')
)