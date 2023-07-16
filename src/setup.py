import setuptools

setuptools.setup(
    name="auto_hgnn",
    version="0.0.1",
    author="Florian Teich",
    author_email="florianteich@gmail.com",
    description="utils",
    long_description="Auto HGNN utils",
    long_description_content_type="text/markdown",
    url="https://github.com/FlorianTeich/hgnn_demos",
    project_urls={
        "Bug Tracker": "https://github.com/FlorianTeich/hgnn_demos/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["auto_hgnn"],
    package_dir={"auto_hgnn": "auto_hgnn"},
    #package_data={'auto_hgnn': ['plugins/*.jar']},
    python_requires=">=3.10",
)