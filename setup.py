from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='trainer',
        packages=find_packages(),
        install_requires=[
            'hyperopt',
            'networkx==1.11',
            'pillow',
            'tf-nightly==1.5.0.dev20171219'
        ]
    )
