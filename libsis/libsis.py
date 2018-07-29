#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2018  Carmelo Mordini et al.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import time
import datetime
import os
from io import BytesIO

def thalammerize(image):
    image += 1
    image = image * (2**16)/10
    image = np.clip(image, 0, 2**16-1)
    return image

def read_sis_header(filename, len=512):
    with open(filename, 'rb') as f:
        head = f.read(8)
        shape = f.read(6)
        timestamp = f.read(20)
        tail = f.read(len-34)
    shape = np.frombuffer(shape, np.uint16).astype(int)
    if shape[0] == 1 or shape[0] == 12336:
        shape = shape[1:]
    return head, shape, timestamp, tail



def read_sis(filename, verbose=False, full_output=False):
    ''' Read sis files, both old version and SisV2

    Parameters
    ----------
    filename : stringa con il nome o il path relativo del file.

    Returns
    -------
    im1 : ndarray 2D
        first half of the image [righe 0 : height/2-1].
    im2 : ndarray 2D
        second half of the image [righe height/2 : height-1].
    image : ndarray 2D
        whole image
    rawdata : ndarray 1D
        raw data read from file
    stringa : string
        comment and datestamp of the image
    block : tuple (Bheight,Bwidth)
        Bheight : int
            y dimension of the sub-block
        Bwidth : int
            x dimension of the sub-block
    Notes
    -----
    NB all the outputs are slices of the raw data: any modifications of them will be reflected also in the linked rawdata elements
    '''
    head, shape, timestamp, tail = read_sis_header(filename, len=512)

    if verbose:
        print("Opening %s"%filename)
    # Sometimes it gives error if the file is opened a sigle time

    f = open(filename, 'rb')                        # open in reading and binary mode
    rawdata = np.fromfile(f,np.uint16).astype(np.int)
    f.close()

    # Dimension of the whole image
    height, width = shape

    # Reading the images
    image = rawdata[-width*height:]
    image.resize(*shape).atype(np.float)

    # defines the two images in the sis
    if full_output:
        im0 = image[:height//2, :]
        im1 = image[height//2:, :]
        return im0, im1, shape, timestamp
    else:
        return image

def write_sis_header(fid, shape, len=512):
    head = 'SisV2' + '.' + '0'*2 #len = 8
    fid.write(head.encode())

    #shape
    height, width = shape
    depth = 1
    size = np.array([depth, height, width], dtype=np.uint16) #len = 6
    fid.write(bytes(size))

    # timestamp
    ts = time.time()
    phead = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-T%H:%M:%S') #len = 20
    fid.write(phead.encode()) # len = 20

    # More: commitProg + descriptive stamp
    freeHead = '0'*(len-34)
    fid.write(freeHead.encode())



def write_sis(filename, image, sisposition=None, thalammer=True):
    """
    Low-level interaction with the sis file for writing it.
    Writes the whole image, with the unused part filled with zeros.

    Args:
        image (np.array): the 2d-array that must be writed after conversion
        to 16-bit unsigned-integers (must be already normalized)
        filename (string): sis filename
        Bheight (int): the y dimension of the eventual block
        Bwidth (int): the x dimension of the eventual block
        stamp (string): a string to describe who, why and what you want
    """
    #keep the double-image convention for sis files, filling the unused
    #with zeros
    if sisposition == 'single':
        image = image
    elif sisposition == 0:
        image = np.concatenate((image, np.zeros_like(image)))
    elif sisposition == 1:
        image = np.concatenate((np.zeros_like(image), image))
    elif sisposition is None:
        image = np.concatenate((image, image))

    with BytesIO() as fid:
        # Write here SisV2 + other 4 free bytes
        write_sis_header(fid, image.shape, len=512)

        if thalammer:
            image = thalammerize(image)

        fid.write(bytes(image.astype(np.uint16)))

        sis_writeOUT(filename, fid.getvalue())

    print('sis written to ' + filename)

def sis_writeOUT(filename, binData):
    """
    Writing of .sis file using a placeholder tmp file, to avoid reading of
    incomplete files when transfer speed is low
    """
    with open(filename+".tmp", 'w+b') as fid:
        fid.write(binData)
    os.rename(filename+".tmp", filename)
