from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 1
_version_minor = 0
_version_micro = ''  # use '' for first of series, number for 1 and above
#_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: GNU GPL v3',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Topic :: Scientific/Engineering']

# Description should be a one-liner:
description = 'libsis: a Python unified library for reading-writing .sis image files'
# Long description will go up on the pypi page
long_description = '''
libsis
========
``libsis`` is a numpy-based Python library for reading-writing .sis image files

License
=======
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright (c) 2018 Carmelo Mordini
'''

NAME = 'libsis'
MAINTAINER = 'Carmelo Mordini'
MAINTAINER_EMAIL = 'carmelo.mordini@unitn.com'
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = 'https://github.com/BEC-Trento/libsis'
DOWNLOAD_URL = ''
LICENSE = 'GNU GPL v3'
AUTHOR = 'Carmelo Mordini'
AUTHOR_EMAIL = 'carmelo.mordini@unitn.com'
PLATFORMS = 'OS Independent'
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['libsis',
            ]
PACKAGE_DATA = {'libsis': [pjoin('data', '*')]}
REQUIRES = ['numpy']
