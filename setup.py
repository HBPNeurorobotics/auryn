import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erbp_utils",
    version="0.0.1",
    author="Alexander Friedrich",
    author_email="friedric@fzi.de",
    description="Python helpers for event-based random backpropagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://ids-git.fzi.de/hbp/erbp",
    packages=["erbp_utils"]
)
