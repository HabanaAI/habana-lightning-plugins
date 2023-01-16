# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from setuptools import setup, Extension
from pathlib import Path
import os
import sys

root = os.environ["HABANA_LIGHTNING_PLUGINS_ROOT"]
REQUIREMENTS = [
    'pytorch-lightning',
    'torch',
    'torchvision',
    'habana-torch-dataloader',
    'habana-torch-plugin',
]

def get_version():
    HABANA_DEFAULT_VERSION = "0.0.0.0"
    version = os.getenv('RELEASE_VERSION')
    if version:
        build_number = os.getenv('RELEASE_BUILD_NUMBER')
        if build_number:
            return version + '.' + build_number
        else:
            return version + '.0'
    else:
        try:
            import subprocess
            import re
            describe = (
                subprocess.check_output(
                    ["git", "-C", root, "describe", "--abbrev=7", "--tags", "--dirty", "--always"])
                .decode("ascii").strip())
            sha = re.search(r"g([a-z0-9\-]+)", describe).group(1)
            return HABANA_DEFAULT_VERSION + "+" + sha
        except Exception as e:
            print("Error getting version: {}".format(e), file=sys.stderr)
            return f"{HABANA_DEFAULT_VERSION}+unknown"

setup(name='habana-lightning-plugins',
      version=get_version(),
      description="Habana's lightning-specific optimized plugins",
      url="https://habana.ai/",
      download_url="TODO",
      license="See LICENSE file",
      license_files=("LICENSE",),
      author="Habana Labs Ltd., an Intel Company",
      author_email="support@habana.ai",
      install_requires=REQUIREMENTS,
      zip_safe=False,
      packages=["habana_lightning_plugins"],
      )
