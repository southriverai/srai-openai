from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="srai-openai",
    packages=find_packages(),
    version="0.5.0",
    license="MIT",
    package_data={},
    python_requires=">=3.5",
    install_requires=requirements,
    author="Jaap Oosterbroek",
    author_email="jaap.oosterbroek@southriverai.com",
    description="A set of functions to better interact with openai.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/southriverai/srai-openai",
    download_url="https://github.com/southriverai/srai-openai/archive/v_01.tar.gz",
    keywords=["SRAI", "TOOLS"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
