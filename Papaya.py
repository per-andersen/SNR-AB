#Written by Martin Sparre, June 2016.
import numpy
import h5py

def WritePapayaHDF5(Filename,Dict,Attrs={}):
    'Dict is a dictionary containing numpy-arrays. Attrs an optional Dict with e.g. header information.'

    F = h5py.File(Filename,'w')

    for key in Dict.keys():
        dset = F.create_dataset(key, Dict[key].shape, dtype=Dict[key].dtype)
        dset[:] = Dict[key]

    dset = F.create_dataset('Attrs', (0,), dtype=None)
    for key in Attrs.keys():
        dset.attrs[key] = Attrs[key]

    F.close()

def ReadPapayaHDF5(Filename):
    'input Filename previosly saved file. Outputs Dict and Attrs'

    F = h5py.File(Filename,'r')

    Dict = {}
    for key in F.keys():
        if key != 'Attrs':
            Dict[key] = F[key].value

    Attrs = {}
    for key in F['Attrs'].attrs.keys():
            Attrs[key] = F['Attrs'].attrs.get(key)

    F.close()
    return Dict,Attrs
