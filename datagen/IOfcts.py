import numpy as np
import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import xml.etree.ElementTree as ET

import os, sys
import struct

"""
Read vtk files (AMITEX output)
"""
def vtkFieldReader(vtk_name, fieldName='tomo_Volume'):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_name)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    siz = list(dim)
    siz = [i - 1 for i in siz]
    orig = data.GetOrigin()
    spacing = data.GetSpacing()
    mesh = vtk_to_numpy(data.GetCellData().GetArray(fieldName))
    return mesh.reshape(siz, order='F'), orig, spacing


'''
Read vti files (velocity field from the post-proc of AMITEX output)
'''
def vtiFieldReader(vti_name, components=[0,1,2]):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_name)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    siz = list(dim)
    nvx = siz[0]*siz[1]*siz[2]
    
    orig = data.GetOrigin()
    spacing = data.GetSpacing()
    
    ncomps = data.GetPointData().GetNumberOfComponents()
    scalars = data.GetPointData().GetScalars()
    scalarsName = scalars.GetName()
    
    v = list()
    for component in components:
        tmp = np.array([scalars.GetComponent(i,component) for i in range(nvx)]) \
                         .astype(scalars.GetDataTypeAsString())
        v.append( np.moveaxis(tmp.reshape(siz), 0, 1) )
    
    return v




'''
Extract parameters/coefficents from mat*.xml file
    Inputs
    ------
        > fname : file name
        >   mID : (default 1) material ID for the porous solid region
        >   idx : (default 1) indices to be extracted, can be a 1D array or scalar
    Outputs
    -------
        > coeff : coefficients to be extracted
'''
#def extract_mat(fname, mID=1, idx=1):
#    
#    tree = ET.parse(fname)
#    root = tree.getroot()
#    
#    #print(' Material ID: ' + root[mID].attrib['numM'])
#    #print(' --- coeff indices: ' + str(idx))
#    
#    if np.isscalar(idx):
#        if root[mID][idx].attrib['Type']=="Constant":
#            coeff = float(root[mID][idx].attrib['Value'])
#        elif root[mID][idx].attrib['Type']=="Constant_Zone":
#            coeff = read_bin(root[mID][idx].attrib['File'])
#    else:
#        coeff = list()
#        for i in idx:
#            if root[mID][i].attrib['Type']=="Constant":
#                coeff.append(float(root[mID][i].attrib['Value']))
#            elif root[mID][i].attrib['Type']=="Constant_Zone":
#                coeff.append(read_bin(root[mID][i].attrib['File']))
#    
#    coeff = np.moveaxis(np.array(coeff), 0, -1)
#    
#    return coeff
def extract_mat(fname, mID=1, idx=1, relativePath=False):
    
    tree = ET.parse(fname)
    root = tree.getroot()
    
    #print(' Material ID: ' + root[mID].attrib['numM'])
    #print(' --- coeff indices: ' + str(idx))
    
    #reference material paramters: Lambda0, Mu0
    if mID == 'REFERENCE':
        ref_mat = root.find('Reference_Material')
        return ref_mat.get('Lambda0'), ref_mat.get('Mu0')
    
    #coefficients
    if np.isscalar(idx):
        if root[mID][idx].attrib['Type']=="Constant":
            coeff = float(root[mID][idx].attrib['Value'])
        elif root[mID][idx].attrib['Type']=="Constant_Zone":
            if relativePath == True:
                if os.name == 'nt':
                    binname = os.path.join(*(fname.split('\\')[:-1] + root[mID][idx].attrib['File'].split('/')[-2:]))
                elif os.name == 'posix':
                    binname = os.path.join(*(fname.split('/')[:-1] + root[mID][idx].attrib['File'].split('/')[-2:]))
                    binname = '/' + binname
            else:
                binname = root[mID][idx].attrib['File']
            coeff = read_bin(binname)
    else:
        coeff = list()
        for i in idx:
            if root[mID][i].attrib['Type']=="Constant":
                coeff.append(float(root[mID][i].attrib['Value']))
            elif root[mID][i].attrib['Type']=="Constant_Zone":
                if relativePath == True:
                    if os.name == 'nt':
                        binname = os.path.join(*(fname.split('\\')[:-1] + root[mID][i].attrib['File'].split('/')[-2:]))
                    elif os.name == 'posix':
                        binname = os.path.join(*(fname.split('/')[:-1] + root[mID][i].attrib['File'].split('/')[-2:]))
                        binname = '/' + binname
                else:
                    binname = root[mID][i].attrib['File']
                coeff.append(read_bin(binname))
                
    coeff = np.moveaxis(np.array(coeff), 0, -1)
    
    return coeff
    
       


'''
Extract parameters from algo*.xml file
    Inputs
    ------
        > fname : file name
        >   tag : e.g. "Non_local_algorithm"
        >   idx : indices to be extracted, can be a 1D array or scalar
    Outputs
    -------
        > param : parameters to be extracted
'''
def extract_algo(fname, tag, idx):

    tree = ET.parse(fname)
    root = tree.getroot()

    for child in root:
        if child.tag==tag:
            if np.isscalar(idx):
                param = child[idx].attrib['Value']
            else:
                param = list()
                for i in idx:
                    param.append(float(child[i].attrib['Value']))
                param = np.array(param)
                
    return param



"""
#file name *.vti
#volume data (3D, 4D array, shape: [x,y,z] or [x,y,z,c])
#volume name, e.g. velocity, phase, etc.
#origin
#spacing in x,y,z direction, (3x1 array)
#number of voxels in x,y,z direction, (3x1 array)
"""
def saveField2VTK(fileout, vdata, vname, origin=[0,0,0], spacing=[1,1,1], Legacy=None):
    
    #
    dx, dy, dz = spacing
    x0, y0, z0 = origin
    vcomponents = np.array(vdata[0,0,0]).size
    
    #dimension + swap components in case of vector input    
    if vcomponents==1:
        nx, ny, nz = np.shape(vdata)
    elif vcomponents==3:
        nx, ny, nz = np.shape(vdata[:,:,:,0])
    
    #swap x and z axes
    vdata = np.swapaxes(vdata, 0, 2)
    
    #data type
    vtype = vtk.util.numpy_support.get_vtk_array_type(vdata.dtype)
    
    #create vtk image object
    imageData = vtk.vtkImageData()
    imageData.SetSpacing(dx, dy, dz)
    imageData.SetOrigin(x0, y0, z0)
    imageData.SetDimensions(nx, ny, nz)
    imageData.AllocateScalars(vtype, vcomponents)
    
    vtk_data_array = numpy_to_vtk(num_array=vdata.ravel(), deep=True, array_type=vtype)
    vtk_data_array.SetNumberOfComponents(vcomponents)
    vtk_data_array.SetName(vname)
    imageData.GetPointData().SetScalars(vtk_data_array)
    
    if Legacy==None or Legacy==False:
        writer = vtk.vtkXMLImageDataWriter()
    elif Legacy==True:
        writer = vtk.vtkStructuredPointsWriter()
        writer.SetFileTypeToBinary()
        
    
    writer.SetInputData(imageData)
    writer.SetFileName(fileout)
    writer.Write()




'''
Identify the corresponding data type of a numpy array
'''
def dtype_numpy2vtk(vdata):
    if vdata.dtype == 'uint8':
        vtktype = 'unsigned_char'
    elif vdata.dtype == 'uint16':
        vtktype = 'unsigned_short'
    elif vdata.dtype == 'uint32':
    
        vtktype = 'unsigned_int'
    elif vdata.dtype == 'uint64':
        vtktype = 'unsigned_long'
    elif vdata.dtype == 'float32':
        vtktype = 'float'
    elif vdata.dtype == 'float64':
        vtktype = 'double'
    else:
        raise TypeError("data type not supported!")
    
    return vtktype


'''
Save mesh image to VTK file specified by AMITEX
'''
def saveMesh2VTK_amitex(fileout, vdata, vname, origin=[0,0,0], spacing=[1,1,1]):
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    
    if len(np.shape(vdata))==2:
        vdata = np.expand_dims(vdata, axis=2)
        
    nx, ny, nz = np.shape(vdata)
    
    vtktype = dtype_numpy2vtk(vdata)
    
    #swap x and z axes
    vdata = np.swapaxes(vdata, 0, 2).copy(order='C')
    
    #change to BigEndian (requirement by VTK for Legacy binary data)
    if sys.byteorder=='little':
        vdata = vdata.byteswap()
    
    #write header
    with open(fileout, 'w') as f:
        f.write("# vtk DataFile Version 4.2\n")
        f.write("mesh_grid\n")
        f.write("BINARY\n")
        f.write("DATASET {}\n".format("STRUCTURED_POINTS"))
        f.write("DIMENSIONS {:d} {:d} {:d}\n".format(nx+1, ny+1, nz+1))
        f.write("ORIGIN {:e} {:e} {:e}\n".format(x0, y0, z0))
        f.write("SPACING {:e} {:e} {:e}\n".format(dx, dy, dz))
        f.write("CELL_DATA {:d}\n".format(nx*ny*nz))
        f.write("SCALARS {} {}\n".format(vname, vtktype))
        f.write("LOOKUP_TABLE {}\n".format("default"))
    
    #write data
    with open(fileout, 'ab') as f:
        f.write(vdata)
        #vdata.tofile(f)


'''
Use this function to properly format the XML file
'''
def xml_indent(elem, level=0):
#code from internet (not verified)
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xml_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


'''
Write material parameters into XML file
    - preparation for AMITEX
'''
def write_AMITEX_xml_mat(xmlname, Lambda0, Mu0, matlib, matlaw, coeffs, intvars=None, relativePath=None):
    
    root = ET.Element("Materials")
    
    nMAT = len(coeffs)
    ncoeffs = len(coeffs[0])
    if intvars is not None:
        nvars = len(intvars[0])
    
    # Reference material
    refmat = ET.Element("Reference_Material")
    refmat.set("Lambda0", str(Lambda0))
    refmat.set("Mu0", str(Mu0))
    root.append(refmat)
    
    # windows or linux
    if os.name=='nt':
        delimiter = '\\'
    elif os.name=='posix':
        delimiter = '/'
        
    # Material
    for imat in range(nMAT):
        
        # material attributes (num, Lib, Law)
        child = ET.Element("Material")
        child.set("numM", str(imat+1))
        child.set("Lib", matlib)
        child.set("Law", matlaw)
        
        # coefficients
        for icoeff in range(ncoeffs):
            cchild = ET.Element("Coeff")
            cchild.set("Index", str(icoeff+1))
            if isinstance(coeffs[imat][icoeff], str):
                cchild.set("Type", "Constant_Zone")
                cchild.set("File", coeffs[imat][icoeff])
                cchild.set("Format", "binary")
            elif np.isscalar(coeffs[imat][icoeff]):
                cchild.set("Type", "Constant")
                cchild.set("Value", str(coeffs[imat][icoeff]))
            else:
                cchild.set("Type", "Constant_Zone")
                tmp_cwd = os.getcwd()
                os.chdir(xmlname[:xmlname.rfind(delimiter)])
                tmp_path = os.path.join(os.getcwd(), xmlname[xmlname.rfind(delimiter)+1:-4])
                if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
                tmp_file = os.path.join(tmp_path, 'mat{}'.format(int(imat)) + '_coeff{}.bin'.format(int(icoeff)))
                os.chdir(tmp_cwd)
                write_bin(tmp_file, coeffs[imat][icoeff])  #coeff needs to be a numpy array
                if relativePath==True:
                    cchild.set("File", os.path.basename(tmp_path) + '/mat{}'.format(int(imat)) + '_coeff{}.bin'.format(int(icoeff)))
                else:
                    cchild.set("File", tmp_file)
                cchild.set("Format", "binary")
            child.append(cchild)
        
        # internal variables (iniital velocity)
        if intvars is not None:
            for ivar in range(nvars):
                cchild = ET.Element("IntVar")
                cchild.set("Index", str(ivar+1))
                if np.isscalar(intvars[imat][ivar]):
                    cchild.set("Type", "Constant")
                    cchild.set("Value", str(intvars[imat][ivar]))
                else:
                    cchild.set("Type", "Constant_Zone")
                    tmp_cwd = os.getcwd()
                    os.chdir(xmlname[:xmlname.rfind(delimiter)])
                    tmp_path = os.path.join(os.getcwd(), xmlname[xmlname.rfind(delimiter)+1:-4])
                    if not os.path.exists(tmp_path):
                        os.mkdir(tmp_path)
                    tmp_file = os.path.join(tmp_path, 'mat{}'.format(int(imat)) + '_intvar{}.bin'.format(int(ivar)))
                    os.chdir(tmp_cwd)
                    write_bin(tmp_file, intvars[imat][ivar])  #intvar needs to be a numpy array
                    if relativePath==True:
                        cchild.set("File", os.path.basename(tmp_path) + '/mat{}'.format(int(imat)) + '_intvar{}.bin'.format(int(ivar)))
                    else:
                        cchild.set("File", tmp_file)
                    cchild.set("Format", "binary")
                child.append(cchild)
        
        
        # Coefficient index for the nonlocal model
        cchild = ET.Element("IndexCoeffNloc")
        cchild.set("NLocMod_num", "1")
        cchild.text = ' '.join([str(x) for x in np.linspace(1, ncoeffs, ncoeffs).astype('int')])
        child.append(cchild)
        
        # Variable index for the nonlocal model
        cchild = ET.Element("IndexVarNloc")
        cchild.set("NLocMod_num", "1")
        if intvars is None:
            cchild.text = " "
        else:
            cchild.text = ' '.join([str(x) for x in np.linspace(1, nvars, nvars).astype('int')])
        child.append(cchild)
        
        root.append(child)
    
    # Nonlocal model setup
    child = ET.Element("Non_local_modeling")
    child.set("NLocMod_num", "1")
    child.set("Modelname", "user_nloc1")
    if intvars is None:
        child.set("Nnloc", "0")
        child.set("Ngnloc", "0")
    else:
        child.set("Nnloc", "3")
        child.set("Ngnloc", "3")
    child.set("Ncoeff_nloc", str(ncoeffs))
    cchild = ET.Element("numM")
    cchild.set("Nmat", str(nMAT))
    cchild.text = ' '.join([str(x) for x in np.linspace(1, nMAT, nMAT).astype('int')])
    child.append(cchild)
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(xmlname, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    #
    print('  --> AMITEX material file: '+xmlname)


'''
Prepare the *.bin file (mat. coefficients) for AMITEX
    input data needs to be numpy array
'''
def write_bin(fname, dat):
    #header: num of points + data type
    with open(fname, 'w') as f:
        f.write("{}\n".format(len(dat)))
        f.write("{}\n".format(dtype_numpy2vtk(dat)))
    #change to BigEndian (requirement by AMITEX)
    if sys.byteorder=='little':
        dat = dat.byteswap()
    #write the binary data
    with open(fname, 'ab') as f:
        f.write(dat)


'''
Read data from the *.bin file (AMITEX use)
'''
def read_bin(fname):
    #read all
    with open(fname,'rb') as f:
        lines = f.readlines()
    #header
    npts = int(lines[0])
    if lines[1].decode('ascii')[:-1]=='double':
        fmt = ">{}d".format(npts)    #the symbol '>' indicates "BigEndian" (required by AMITEX)
        fmt_numpy = np.float64
    elif lines[1].decode('ascii')[:-1]=='float':
        fmt = ">{}f".format(npts)
        fmt_numpy = np.float32
    else:
        raise TypeError("data type not supported! (read_bin)")
    #extract data (numpy array)
    return np.array(struct.unpack(fmt, b''.join(lines[2:]))).astype(fmt_numpy)


'''
Write load parameters into XML file
    - preparation for AMITEX
'''
def write_AMITEX_xml_load(xmlname, outVTKstrain=0, outVTKstress=0):
    
    root = ET.Element("Loading_Output")
    
    # vtk fields to be print (stress/strain/intvar)
    child = ET.Element("Output")
    cchild = ET.Element("vtk_StressStrain")
    cchild.set("Strain", str(outVTKstrain))
    cchild.set("Stress", str(outVTKstress))
    child.append(cchild)
    root.append(child)
    
    # User field initialisation (TODO)
    
    # User sub-field initialisation (TODO)
    
    # successive loading (hard-coded, no user-defined param yet)
    child = ET.Element("Loading")
    child.set("Tag", "1")
    
    cchild = ET.Element("Time_Discretization")
    cchild.set("Discretization", "linear")
    cchild.set("Nincr", "1")
    cchild.set("Tfinal", "1")
    child.append(cchild)
    
    cchild = ET.Element("Output_vtkList")
    cchild.text = "1"
    child.append(cchild)
    
    cchild = ET.Element("xx")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("yy")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("zz")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("xy")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("xz")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    cchild = ET.Element("yz")
    cchild.set("Driving", "Strain")
    cchild.set("Evolution", "Linear")
    cchild.set("Value", "0")
    child.append(cchild)
    
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(xmlname, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    #
    print('  --> AMITEX load file: '+xmlname)



'''
Write algorithm parameters into XML file
    - preparation for AMITEX
'''
def write_AMITEX_xml_algo(xmlname, params):
    
    nparams = len(params)
    
    root = ET.Element("Algorithm_Parameters")
    
    # Algorithm (amitex core algo -- hard coded)
    child = ET.Element("Algorithm")
    child.set("Type", "Basic_Scheme")
    cchild = ET.Element("Convergence_Criterion")
    cchild.set("Value", "Default")
    child.append(cchild)
    cchild = ET.Element("Convergence_Acceleration")
    cchild.set("Value", "False")
    child.append(cchild)
    cchild = ET.Element("Nitermin")
    cchild.set("Value", "0")
    child.append(cchild)    
    root.append(child)
    
    # Mechanics (hard coded)
    child = ET.Element("Mechanics")
    cchild = ET.Element("Filter")
    cchild.set("Type", "Default")
    child.append(cchild)
    cchild = ET.Element("Small_Perturbations")
    cchild.set("Value", "True")
    child.append(cchild)
    root.append(child)
    
    # Nonlocal model
    child = ET.Element("Non_local_algorithm")
    child.set("NLocMod_num", "1")
    child.set("Algo", "explicit")
    
    for iparam in range(nparams):
        cchild = ET.Element("P_real")
        cchild.set("Index", str(iparam+1))
        cchild.set("Value", str(params[iparam]))
        child.append(cchild)
    
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(xmlname, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    
    #
    print('  --> AMITEX algo file: '+xmlname)


'''
Write the launcher script for AMITEX
'''
def write_AMITEX_launcher(scriptname, prefix, env_amitex="../../../../amitex_fftp/env_amitex.sh",\
                                              exe_amitex="../../src/amitex_fftp", \
                                              prefix_load="res", \
                                              cpus=None):
    with open(scriptname, 'w') as f:
        f.write("#!/bin/bash\n")
        
        f.write("source {}\n".format(env_amitex))
        f.write("AMITEX0={}\n".format('"'+exe_amitex+'"'))
        
        f.write("MESH={}\n".format('"'+'../micro/'+prefix+'_mesh.vtk'+'"'))
        f.write("ALGO={}\n".format('"'+'../algo/'+prefix+'_algo.xml'+'"'))
        f.write("MATE={}\n".format('"'+'../mate/'+prefix+'_mat.xml'+'"'))
        f.write("LOAD={}\n".format('"'+'../load/'+prefix+'_load.xml'+'"'))
        
        f.write("mkdir ../results\n")
        f.write("mkdir ../results/{}\n".format(prefix))
        
        if cpus==None:
            f.write("mpirun $AMITEX0 -nm $MESH -m $MATE -c $LOAD -a $ALGO -s ../results/{}/{}\n" \
                    .format(prefix, prefix_load))
        else:
            f.write("mpirun -np {:d} $AMITEX0 -nm $MESH -m $MATE -c $LOAD -a $ALGO -s ../results/{}/{}\n" \
                    .format(cpus, prefix, prefix_load))
                
def write_AMITEX_launcher_zID(scriptname, prefix, env_amitex="../../../../amitex_fftp/env_amitex.sh",\
                                                  exe_amitex="../../src/amitex_fftp", \
                                                  prefix_load="res", \
                                                  cpus=None):
    with open(scriptname, 'w') as f:
        f.write("#!/bin/bash\n")
        
        f.write("source {}\n".format(env_amitex))
        f.write("AMITEX0={}\n".format('"'+exe_amitex+'"'))
        
        f.write("MESH={}\n".format('"'+'../micro/'+prefix+'_mesh.vtk'+'"'))
        f.write("ZONE={}\n".format('"'+'../micro/'+prefix+'_zID.vtk'+'"'))
        f.write("ALGO={}\n".format('"'+'../algo/'+prefix+'_algo.xml'+'"'))
        f.write("MATE={}\n".format('"'+'../mate/'+prefix+'_mat.xml'+'"'))
        f.write("LOAD={}\n".format('"'+'../load/'+prefix+'_load.xml'+'"'))
        
        f.write("mkdir ../results\n")
        f.write("mkdir ../results/{}\n".format(prefix))
        
        if cpus==None:
            f.write("mpirun $AMITEX0 -nm $MESH -nz $ZONE -m $MATE -c $LOAD -a $ALGO -s ../results/{}/{}\n" \
                .format(prefix, prefix_load))
        else:
            f.write("mpirun -np {:d} $AMITEX0 -nm $MESH -nz $ZONE -m $MATE -c $LOAD -a $ALGO -s ../results/{}/{}\n" \
                .format(cpus, prefix, prefix_load))



def write_AMITEX_launcher_ml(scriptname, \
                             meshname=None, \
                             zoneIDname=None, \
                             algoname=None, \
                             matename=None, \
                             loadname=None, \
                             resuname=None, \
                             env_amitex="../../../../amitex_fftp/env_amitex.sh",\
                             exe_amitex="../../src/amitex_fftp",
                             cpus=None, \
                             prefix="vmacro"):
    with open(scriptname, 'w') as f:
        f.write("#!/bin/bash\n")
        
        f.write("source {}\n".format(env_amitex))
        f.write("AMITEX0={}\n".format('"'+exe_amitex+'"'))
        
        if meshname:
            f.write("MESH={}\n".format('"'+meshname+'"'))
        if zoneIDname:
            f.write("ZONE={}\n".format('"'+zoneIDname+'"'))
        if algoname:
            f.write("ALGO={}\n".format('"'+algoname+'"'))
        if matename:
            f.write("MATE={}\n".format('"'+matename+'"'))
        if loadname:
            f.write("LOAD={}\n".format('"'+loadname+'"'))
        if resuname:
            f.write("RESU={}\n".format('"'+resuname+'"'))
            f.write("mkdir {}\n".format(resuname))
        
        if cpus is None:
            mpirun_command = "mpirun $AMITEX0"
        else:
            mpirun_command = "mpirun -np {:d} $AMITEX0".format(cpus)

        if meshname:
            mpirun_command += " -nm $MESH"
        if zoneIDname:
            mpirun_command += " -nz $ZONE"
        if matename:
            mpirun_command += " -m $MATE"
        if loadname:
            mpirun_command += " -c $LOAD"
        if algoname:
            mpirun_command += " -a $ALGO"
        if resuname:
            mpirun_command += " -s $RESU/{}".format(prefix)
        
        f.write(mpirun_command + "\n")
        
        
'''
Save the collated information (vfield and macroscopic props) into XML file
'''
def saveBrinkman2XML(prefix_output, meshname, vmacro, W, H, idx):
    
    # read algorithm parameters
    mu = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['mu'])
    mue = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['mue'])
    crit = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['crit'])
    FDscheme = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['FD'])
    ACV = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['ACV'])
    modACV = extract_algo(prefix_output+'_algo.xml', 'Non_local_algorithm', idx['modACV'])
    
    #############
    # extract the computation performance (TODO)
    logfile = prefix_output + '.log'
    #############
    
    # write the collated info into an xml file
    root = ET.Element("AMITEX_Brinkman")
    
    # simulation setting
    child = ET.Element("Setting")
    
    cchild = ET.Element("Mesh")
    cchild.set("File", meshname)
    child.append(cchild)
    
    cchild = ET.Element("Viscosity")
    cchild.set("mu", mu)
    cchild.set("mue", mue)
    child.append(cchild)
    
    cchild = ET.Element("FiniteDifferenceScheme")
    cchild.set("Tag", FDscheme)
    child.append(cchild)
    
    cchild = ET.Element("ConvAccelaration")
    cchild.set("activate", ACV)
    cchild.set("increment", modACV)
    child.append(cchild)
    
    cchild = ET.Element("ConvTolerence")
    cchild.set("Value", crit)
    child.append(cchild)
    
    root.append(child)
    
    # simulation result
    child = ET.Element("Result")
    
    cchild = ET.Element("MacroVelocity")
    cchild.set("vx", str(vmacro[0]))
    cchild.set("vy", str(vmacro[1]))
    cchild.set("vz", str(vmacro[2]))
    child.append(cchild)
    
    cchild = ET.Element("MacroPressureGradient")
    cchild.set("Gx", str(W[0]))
    cchild.set("Gy", str(W[1]))
    cchild.set("Gz", str(W[2]))
    child.append(cchild)
    
    cchild = ET.Element("MacroResistivity")
    cchild.set("Hx", str(H[0]))
    cchild.set("Hy", str(H[1]))
    cchild.set("Hz", str(H[2]))
    child.append(cchild)
    
    # computation performance
    cchild = ET.Element("Performance")
    cchild.set("nIters", "")
    cchild.set("wallT", "")
    child.append(cchild)
    
    root.append(child)
    
    # write
    xml_indent(root)
    tree = ET.ElementTree(root)
    with open(prefix_output+'_collated.xml', "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

