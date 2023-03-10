# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, Extension, find_namespace_packages
from pathlib import Path
import os
import sys

root = os.environ.get("HABANA_LIGHTNING_PLUGINS_ROOT", '.')
long_description = Path(os.path.join(root, "README.md")).read_text()

REQUIREMENTS = [
    'pytorch-lightning',
    'torch',
    'torchvision',
    'habana-torch-dataloader',
    'habana-torch-plugin',
]

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: Other/Proprietary License',
    'Operating System :: Unix',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
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
      license="See LICENSE file",
      license_files=("LICENSE",),
      author="Habana Labs Ltd., an Intel Company",
      author_email="support@habana.ai",
      install_requires=REQUIREMENTS,
      zip_safe=False,
      packages=find_namespace_packages(include=[
        "habana_lightning_plugins",
        "habana_lightning_plugins.dataloaders"]),
      classifiers=CLASSIFIERS,
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
