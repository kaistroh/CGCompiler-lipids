"""
gromacstopmodifier.py is largely based on:

gromacstopfile.py: Used for loading Gromacs top files.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2016 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Jason Swails

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import re
import distutils.spawn
from collections import OrderedDict 

try:
    import pwd
    try:
        _username = pwd.getpwuid(os.getuid())[0]
    except KeyError:
        _username = 'username'
    _userid = os.getuid()
    _uname = os.uname()[1]
except ImportError:
    import getpass
    _username = getpass.getuser()   # pragma: no cover
    _userid = 0                     # pragma: no cover
    import platform                 # pragma: no cover
    _uname = platform.node()        # pragma: no cover



novarcharre = re.compile(r'\W')

def _find_all_instances_in_string(string, substr):
    """ Find indices of all instances of substr in string """
    indices = []
    idx = string.find(substr, 0)
    while idx > -1:
        indices.append(idx)
        idx = string.find(substr, idx+1)
    return indices
    

def _replace_defines(line, defines):
    """ Replaces defined tokens in a given line """
    if not defines: return line
    for define in reversed(defines):
        value = defines[define]
        indices = _find_all_instances_in_string(line, define)
        if not indices: continue
        # Check to see if it's inside of quotes
        inside = ''
        idx = 0
        n_to_skip = 0
        new_line = []
        for i, char in enumerate(line):
            if n_to_skip:
                n_to_skip -= 1
                continue
            if char in ('\'"'):
                if not inside:
                    inside = char
                else:
                    if inside == char:
                        inside = ''
            if idx < len(indices) and i == indices[idx]:
                if inside:
                    new_line.append(char)
                    idx += 1
                    continue
                if i == 0 or novarcharre.match(line[i-1]):
                    endidx = indices[idx] + len(define)
                    if endidx >= len(line) or novarcharre.match(line[endidx]):
                        new_line.extend(list(value))
                        n_to_skip = len(define) - 1
                        idx += 1
                        continue
                idx += 1
            new_line.append(char)
        line = ''.join(new_line)

    return line
    
    
class GromacsTopFile(object):
    """GromacsTopFile parses a Gromacs top file and constructs a Topology and (optionally) an OpenMM System from it."""

    class _MoleculeType(object):
        """Inner class to store information about a molecule type."""
        def __init__(self):
            self.atoms = []
            self.bonds = []
            self.angles = []
            self.dihedrals = []
            self.exclusions = []
            self.pairs = []
            self.constraints = []
            self.cmaps = []
            self.vsites2 = []
            self.has_virtual_sites = False
            self.has_nbfix_terms = False
            self.nrexcl = 1
            self.atomid_to_atomname = {}
            self.bondname_to_bondndx = {}
            self.anglename_to_anglendx = {}

        def _makeAtomidDict(self):
            for atom in self.atoms:
                self.atomid_to_atomname[atom[0]] = atom[4]

        def _makeBondnameDict(self):
            for i, bond in enumerate(self.bonds):
                bondname = self.atomid_to_atomname[bond[0]] + '-' + self.atomid_to_atomname[bond[1]]
                self.bondname_to_bondndx[bondname] = i

        def _makeAnglenameDict(self):
            for i, angle in enumerate(self.angles):
                atom0 = self.atomid_to_atomname[angle[0]]
                atom1 = self.atomid_to_atomname[angle[1]]
                atom2 = self.atomid_to_atomname[angle[2]]
                anglename = atom0 + "-" + atom1 + '-' + atom2
                self.anglename_to_anglendx[anglename] = i

    def _processFile(self, file):
        append = ''
        for line in open(file):
            if line.strip().endswith('\\'):
                append = '%s %s' % (append, line[:line.rfind('\\')])
            else:
                self._processLine(append+' '+line, file)
                append = ''

    def _processLine(self, line, file):
        """Process one line from a file."""
        if ';' in line:
            line = line[:line.index(';')]
        stripped = line.strip()
        ignore = not all(self._ifStack)
        if stripped.startswith('*') or len(stripped) == 0:
            # A comment or empty line.
            return

        elif stripped.startswith('[') and not ignore:
            # The start of a category.
            if not stripped.endswith(']'):
                raise ValueError('Illegal line in .top file: '+line)
            self._currentCategory = stripped[1:-1].strip()

        elif stripped.startswith('#'):
            # A preprocessor command.
            fields = stripped.split()
            command = fields[0]
            if len(self._ifStack) != len(self._elseStack):
                raise RuntimeError('#if/#else stack out of sync')

            if command == '#include' and not ignore:
                # Locate the file to include
                name = stripped[len(command):].strip(' \t"<>')
                searchDirs = self._includeDirs+(os.path.dirname(file),)
                for dir in searchDirs:
                    file = os.path.join(dir, name)
                    if os.path.isfile(file):
                        # We found the file, so process it.
                        self._processFile(file)
                        break
                else:
                    raise ValueError('Could not locate #include file: '+name)
            elif command == '#define' and not ignore:
                # Add a value to our list of defines.
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top file: '+line)
                name = fields[1]
                valueStart = stripped.find(name, len(command))+len(name)+1
                value = line[valueStart:].strip()
                value = value or '1' # Default define is 1
                self._defines[name] = value
            elif command == '#ifdef':
                # See whether this block should be ignored.
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top file: '+line)
                name = fields[1]
                self._ifStack.append(name in self._defines)
                self._elseStack.append(False)
            elif command == '#undef':
                # Un-define a variable
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top file: '+line)
                if fields[1] in self._defines:
                    self._defines.pop(fields[1])
            elif command == '#ifndef':
                # See whether this block should be ignored.
                if len(fields) < 2:
                    raise ValueError('Illegal line in .top file: '+line)
                name = fields[1]
                self._ifStack.append(name not in self._defines)
                self._elseStack.append(False)
            elif command == '#endif':
                # Pop an entry off the if stack.
                if len(self._ifStack) == 0:
                    raise ValueError('Unexpected line in .top file: '+line)
                del(self._ifStack[-1])
                del(self._elseStack[-1])
            elif command == '#else':
                # Reverse the last entry on the if stack
                if len(self._ifStack) == 0:
                    raise ValueError('Unexpected line in .top file: '+line)
                if self._elseStack[-1]:
                    raise ValueError('Unexpected line in .top file: '
                                     '#else has already been used ' + line)
                self._ifStack[-1] = (not self._ifStack[-1])
                self._elseStack[-1] = True

        elif not ignore:
            # Gromacs occasionally uses #define's to introduce specific
            # parameters for individual terms (for instance, this is how
            # ff99SB-ILDN is implemented). So make sure we do the appropriate
            # pre-processor replacements necessary
            line = _replace_defines(line, self._defines)
            # A line of data for the current category
            if self._currentCategory is None:
                raise ValueError('Unexpected line in .top file: '+line)
            if self._currentCategory == 'defaults':
                self._processDefaults(line)
            elif self._currentCategory == 'moleculetype':
                self._processMoleculeType(line)
            elif self._currentCategory == 'molecules':
                self._processMolecule(line)
            elif self._currentCategory == 'atoms':
                self._processAtom(line)
            elif self._currentCategory == 'bonds':
                self._processBond(line)
            elif self._currentCategory == 'angles':
                self._processAngle(line)
            elif self._currentCategory == 'dihedrals':
                self._processDihedral(line)
            elif self._currentCategory == 'exclusions':
                self._processExclusion(line)
            elif self._currentCategory == 'pairs':
                self._processPair(line)
            elif self._currentCategory == 'constraints':
                self._processConstraint(line)
            elif self._currentCategory == 'cmap':
                self._processCmap(line)
            elif self._currentCategory == 'atomtypes':
                self._processAtomType(line)
            elif self._currentCategory == 'bondtypes':
                self._processBondType(line)
            elif self._currentCategory == 'angletypes':
                self._processAngleType(line)
            elif self._currentCategory == 'dihedraltypes':
                self._processDihedralType(line)
            elif self._currentCategory == 'implicit_genborn_params':
                self._processImplicitType(line)
            elif self._currentCategory == 'pairtypes':
                self._processPairType(line)
            elif self._currentCategory == 'cmaptypes':
                self._processCmapType(line)
            elif self._currentCategory == 'nonbond_params':
                self._processNonbondType(line)
            elif self._currentCategory == 'virtual_sites2':
                self._processVirtualSites2(line)
            elif self._currentCategory.startswith('virtual_sites'):
                if self._currentMoleculeType is None:
                    raise ValueError('Found %s before [ moleculetype ]' %
                                     self._currentCategory)
                self._currentMoleculeType.has_virtual_sites = True

    def _processDefaults(self, line):
        """Process the [ defaults ] line."""
        fields = line.split()
        if len(fields) < 5:
            # fudgeLJ and fudgeQQ not specified, assumed 1.0 by default
            if len(fields) == 3:
                fields.append(1.0)
                fields.append(1.0)
            # generate pairs (third entry) gromacs default: no
            elif len(fields) == 2:
                fields.append('no')
                fields.append(1.0)
                fields.append(1.0)
            else:
                raise ValueError('Too few fields in [ defaults ] line: '+line)
        if fields[0] != '1':
            raise ValueError('Unsupported nonbonded type: '+fields[0])
        if not fields[1] in ('1', '2', '3'):
            raise ValueError('Unsupported combination rule: '+fields[1])
        if fields[2].lower() == 'no':
            self._genpairs = False
        self._defaults = fields

    def _processMoleculeType(self, line):
        """Process a line in the [ moleculetypes ] category."""
        fields = line.split()
        print(fields)
        if len(fields) < 1:
            raise ValueError('Too few fields in [ moleculetypes ] line: '+line)
        type = GromacsTopFile._MoleculeType()
        self._moleculeTypes[fields[0]] = type
        self._currentMoleculeType = type
        type.nrexcl = fields[1]

    def _processMolecule(self, line):
        """Process a line in the [ molecules ] category."""
        fields = line.split()
        if len(fields) < 2:
            raise ValueError('Too few fields in [ molecules ] line: '+line)
        self._molecules.append((fields[0], int(fields[1])))

    def _processAtom(self, line):
        """Process a line in the [ atoms ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ atoms ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ atoms ] line: '+line)
        self._currentMoleculeType.atoms.append(fields)

    def _processBond(self, line):
        """Process a line in the [ bonds ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ bonds ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 3:
            raise ValueError('Too few fields in [ bonds ] line: '+line)
        #if fields[2] != '1':
        # also allow bondtype 6 (harmonic potential, no nb exlusion) used in Martini for 
        # rubber bands.
        if fields[2] not in ('1', '6'):
            raise ValueError('Unsupported function type in [ bonds ] line: '+line)
        self._currentMoleculeType.bonds.append(fields)

    def _processAngle(self, line):
        """Process a line in the [ angles ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ angles ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 4:
            raise ValueError('Too few fields in [ angles ] line: '+line)
        #if fields[3] not in ('1', '5'):
        # added angle type 2: GROMOS-96 simplified function used by Martini
        if fields[3] not in ('1', '2', '5'):
            raise ValueError('Unsupported function type in [ angles ] line: '+line)
        self._currentMoleculeType.angles.append(fields)

    def _processDihedral(self, line):
        """Process a line in the [ dihedrals ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ dihedrals ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ dihedrals ] line: '+line)
        if fields[4] not in ('1', '2', '3', '4', '5', '9'):
            raise ValueError('Unsupported function type in [ dihedrals ] line: '+line)
        self._currentMoleculeType.dihedrals.append(fields)

    def _processExclusion(self, line):
        """Process a line in the [ exclusions ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ exclusions ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 2:
            raise ValueError('Too few fields in [ exclusions ] line: '+line)
        self._currentMoleculeType.exclusions.append(fields)

    def _processPair(self, line):
        """Process a line in the [ pairs ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ pairs ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 3:
            raise ValueError('Too few fields in [ pairs ] line: '+line)
        if fields[2] != '1':
            raise ValueError('Unsupported function type in [ pairs ] line: '+line)
        self._currentMoleculeType.pairs.append(fields)

    def _processConstraint(self, line):
        """Process a line in the [ constraints ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ constraints ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 4:
            raise ValueError('Too few fields in [ constraints ] line: '+line)
        self._currentMoleculeType.constraints.append(fields)

    def _processCmap(self, line):
        """Process a line in the [ cmaps ] category."""
        if self._currentMoleculeType is None:
            raise ValueError('Found [ cmap ] section before [ moleculetype ]')
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ cmap ] line: '+line)
        self._currentMoleculeType.cmaps.append(fields)

    def _processAtomType(self, line):
        """Process a line in the [ atomtypes ] category."""
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ atomtypes ] line: '+line)
        if len(fields[3]) == 1:
            # Bonded type and atomic number are both missing.
            fields.insert(1, None)
            fields.insert(1, None)
        elif len(fields[4]) == 1 and fields[4].isalpha():
            if fields[1][0].isalpha():
                # Atomic number is missing.
                fields.insert(2, None)
            else:
                # Bonded type is missing.
                fields.insert(1, None)
        self._atomTypes[fields[0]] = fields

    def _processBondType(self, line):
        """Process a line in the [ bondtypes ] category."""
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ bondtypes ] line: '+line)
        if fields[2] != '1':
            raise ValueError('Unsupported function type in [ bondtypes ] line: '+line)
        self._bondTypes[tuple(fields[:2])] = fields

    def _processAngleType(self, line):
        """Process a line in the [ angletypes ] category."""
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ angletypes ] line: '+line)
        if fields[3] not in ('1', '5'):
            raise ValueError('Unsupported function type in [ angletypes ] line: '+line)
        self._angleTypes[tuple(fields[:3])] = fields

    def _processDihedralType(self, line):
        """Process a line in the [ dihedraltypes ] category."""
        fields = line.split()
        if len(fields) < 7:
            raise ValueError('Too few fields in [ dihedraltypes ] line: '+line)
        if fields[4] not in ('1', '2', '3', '4', '5', '9'):
            raise ValueError('Unsupported function type in [ dihedraltypes ] line: '+line)
        key = tuple(fields[:5])
        if fields[4] == '9' and key in self._dihedralTypes:
            # There are multiple dihedrals defined for these atom types.
            self._dihedralTypes[key].append(fields)
        else:
            self._dihedralTypes[key] = [fields]

    def _processImplicitType(self, line):
        """Process a line in the [ implicit_genborn_params ] category."""
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('Too few fields in [ implicit_genborn_params ] line: '+line)
        self._implicitTypes[fields[0]] = fields

    def _processPairType(self, line):
        """Process a line in the [ pairtypes ] category."""
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ pairtypes] line: '+line)
        if fields[2] != '1':
            raise ValueError('Unsupported function type in [ pairtypes ] line: '+line)
        self._pairTypes[tuple(fields[:2])] = fields

    def _processCmapType(self, line):
        """Process a line in the [ cmaptypes ] category."""
        fields = line.split()
        if len(fields) < 8 or len(fields) < 8+int(fields[6])*int(fields[7]):
            raise ValueError('Too few fields in [ cmaptypes ] line: '+line)
        if fields[5] != '1':
            raise ValueError('Unsupported function type in [ cmaptypes ] line: '+line)
        self._cmapTypes[tuple(fields[:5])] = fields

    def _processNonbondType(self, line):
        """Process a line in the [ nonbond_params ] category."""
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ nonbond_params ] line: '+line)
        if fields[2] != '1':
            raise ValueError('Unsupported function type in [ nonbond_params ] line: '+line)
        self._nonbondTypes[tuple(sorted(fields[:2]))] = fields

    def _processVirtualSites2(self, line):
        """Process a line in the [ virtual_sites2 ] category."""
        fields = line.split()
        if len(fields) < 5:
            raise ValueError('Too few fields in [ virtual_sites2 ] line: ' + line)
        self._currentMoleculeType.vsites2.append(fields[:5])

    def __init__(self, file, periodicBoxVectors=None,
                             unitCellDimensions=None, 
                             includeDir=None, 
                             defines=None,
                             itp=False):
        """Load a top file.

        Parameters
        ----------
        file : str
            the name of the file to load
        periodicBoxVectors : tuple of Vec3=None
            the vectors defining the periodic box
        unitCellDimensions : Vec3=None
            the dimensions of the crystallographic unit cell.  For
            non-rectangular unit cells, specify periodicBoxVectors instead.
        includeDir : string=None
            A directory in which to look for other files included from the
            top file. If not specified, we will attempt to locate a gromacs
            installation on your system. When gromacs is installed in
            /usr/local, this will resolve to /usr/local/gromacs/share/gromacs/top
        defines : dict={}
            preprocessor definitions that should be predefined when parsing the file
         """
        self._itp = itp 
        #if itp:
        #    raise ValueError('itp files can be read, but current implementation needs out.top')
         
        if includeDir is None:
            includeDir = _defaultGromacsIncludeDir()
        self._includeDirs = (os.path.dirname(file), includeDir)
        # Most of the gromacs water itp files for different forcefields,
        # unless the preprocessor #define FLEXIBLE is given, don't define
        # bonds between the water hydrogen and oxygens, but only give the
        # constraint distances and exclusions.
        self._defines = OrderedDict()
        self._defines['FLEXIBLE'] = True
        self._genpairs = True
        if defines is not None:
            for define, value in defines.iteritems():
                self._defines[define] = value

        # Parse the file.

        self._currentCategory = None
        self._ifStack = []
        self._elseStack = []
        self._moleculeTypes = {}
        #self._scaleMolTypes = None
        self._molecules = []
        self._currentMoleculeType = None
        self._atomTypes = {}
        #self._hotAtoms = None
        self._writeBState = False
        self._bondTypes= {}
        self._angleTypes = {}
        self._dihedralTypes = {}
        self._implicitTypes = {}
        self._pairTypes = {}
        self._cmapTypes = {}
        self._nonbondTypes = {}
        self._processFile(file)
        
        
         
    def _writeMoleculeType(self, dest, molkey):
        mtype = self._moleculeTypes[molkey]         
        
        # Moltype section
        dest.write('[ moleculetype ]\n')
        dest.write('; molname    nrexcl\n')
        dest.write('%-6s    %d\n\n' %(molkey, int(mtype.nrexcl)))

        ## Atoms of Moltype
        dest.write('[ atoms ]\n')
        dest.write('; id     type   resnr  resname atomname cgnr charge ')
        dest.write('\n')
        
        for atom in mtype.atoms:
            #unpack atom
            atomid = int(atom[0])
            atomtype = atom[1]
            resid = int(atom[2])
            resname = atom[3]
            atomname = atom[4]
            cgnr = int(atom[5])
            charge = float(atom[6])
            
            dest.write('%4d %8s %6d %6s %6s %6d  %8.5f' 
                        %(atomid, atomtype, resid,
                          resname, atomname, cgnr, charge) )      
            dest.write('\n')    


        dest.write('\n')
        
        ### Bonds
        if len(mtype.bonds) > 0: 
            dest.write('[ bonds ]\n')
            dest.write(';   i    j    funct   r0   fc\n')
            for bond in mtype.bonds:
                ai = int(bond[0])
                aj = int(bond[1])
                funct = int(bond[2])
                r0 = float(bond[3])
                fc = float(bond[4])
                
                dest.write('%5d %5d %3d   %7.5f %6.4g\n' %(ai, aj, funct, r0, fc))
                
            dest.write('\n')
                
        ### Constraints
        if len(mtype.constraints) > 0:
            dest.write('[ constraints ]\n')
            dest.write(';   i    j    funct   r0\n')
            for constraint in mtype.constraints:
                ai = int(constraint[0])
                aj = int(constraint[1])
                funct = int(constraint[2])
                r0 = float(constraint[3])
                
                dest.write('%5d %5d %3d   %7.5f\n' %(ai, aj, funct, r0))

            dest.write('\n')
        
        ### Angles
        if len(mtype.angles) > 0:
            dest.write('[ angles ]\n')
            dest.write('; i   j   k   funct   theta0  fc\n')
            for angle in mtype.angles:
                ai = int(angle[0])
                aj = int(angle[1])
                ak = int(angle[2])
                funct = int(angle[3])
                theta0 = float(angle[4])
                fc = float(angle[5])
                dest.write('%5d %5d %5d %3d   %7.3f %6.4g\n' 
                            %(ai, aj, ak, funct, theta0, fc))
                            
            dest.write('\n')
            
        ### Dihedrals    
        if len(mtype.dihedrals) > 0:
            dest.write('[ dihedrals ]\n')
            dest.write('; ai aj ak al funct phi k  mult\n')
            for dihedral in mtype.dihedrals:
                    # proper dihedrals
                    if len(dihedral) == 8:
                        ai = int(dihedral[0])
                        aj = int(dihedral[1])
                        ak = int(dihedral[2])
                        al = int(dihedral[3])
                        funct = int(dihedral[4])
                        phi = float(dihedral[5])
                        fc = float(dihedral[6])
                        mult = int(dihedral[7])
                        dest.write('%5d %5d %5d %d %3d   %7.3f %6.4g %3d\n' 
                                    %(ai, aj, ak, al, funct, phi, fc, mult))
                    # improper dihedrals
                    if len(dihedral) == 7:
                        ai = int(dihedral[0])
                        aj = int(dihedral[1])
                        ak = int(dihedral[2])
                        al = int(dihedral[3])
                        funct = int(dihedral[4])
                        phi = float(dihedral[5])
                        fc = float(dihedral[6])
                        dest.write('%5d %5d %5d %d %3d   %7.3f %6.4g\n' 
                                    %(ai, aj, ak, al, funct, phi, fc))
                                    
            dest.write('\n')
            
            
        if len(mtype.pairs) > 0:
            raise ValueError('writing pairs not implemented yet')
            
        if len(mtype.cmaps) > 0:
            raise ValueError('writing cmaps not implemented yet')
            
        if len(mtype.vsites2) > 0:
            raise ValueError('writing vsites2 not implemented yet')
            
        if len(mtype.exclusions) > 0:
            raise ValueError('writing exclusions not implemented yet')

    def changeAtomType(self, molkey, atomname, atomtype_new):
        print(atomname)
        atoms = self._moleculeTypes[molkey].atoms
        for atom in atoms:
            if atom[4] == atomname:
                print(atom[1])
                atom[1] = atomtype_new

    def changeBond(self, molkey, bondname, b0_new, fc_new):
        self._moleculeTypes[molkey]._makeAtomidDict()
        self._moleculeTypes[molkey]._makeBondnameDict()
        bondndx = self._moleculeTypes[molkey].bondname_to_bondndx[bondname]
        self._moleculeTypes[molkey].bonds[bondndx][3] = str(b0_new)
        self._moleculeTypes[molkey].bonds[bondndx][4] = str(fc_new)

    def changeAngle(self, molkey, anglename, a0_new, fc_new):
        self._moleculeTypes[molkey]._makeAtomidDict()
        self._moleculeTypes[molkey]._makeAnglenameDict()
        anglendx = self._moleculeTypes[molkey].anglename_to_anglendx[anglename]
        self._moleculeTypes[molkey].angles[anglendx][4] = str(a0_new)
        self._moleculeTypes[molkey].angles[anglendx][5] = str(fc_new)
        
            
            
    def write(self, dest):
        #own_handle = False

        dest = open(dest, 'w')
        # own_handle = True
        # if not hasattr(dest, 'write'):
        #     raise TypeError('dest must be a file name or file-like object')
        
        #try:               
        #### Moleculetypes
        for key in self._moleculeTypes:
            self._writeMoleculeType(dest, key)
            
        #finally:
            #if own_handle:
        dest.close()

def _defaultGromacsIncludeDir():
    """Find the location where gromacs #include files are referenced from, by
    searching for (1) gromacs environment variables, (2) for the gromacs binary
    'pdb2gmx' or 'gmx' in the PATH, or (3) just using the default gromacs
    install location, /usr/local/gromacs/share/gromacs/top """
    if 'GMXDATA' in os.environ:
        return os.path.join(os.environ['GMXDATA'], 'top')
    if 'GMXBIN' in os.environ:
        return os.path.abspath(os.path.join(os.environ['GMXBIN'], '..', 'share', 'gromacs', 'top'))

    pdb2gmx_path = distutils.spawn.find_executable('pdb2gmx')
    if pdb2gmx_path is not None:
        return os.path.abspath(os.path.join(os.path.dirname(pdb2gmx_path), '..', 'share', 'gromacs', 'top'))
    else:
        gmx_path = distutils.spawn.find_executable('gmx')
        if gmx_path is not None:
            return os.path.abspath(os.path.join(os.path.dirname(gmx_path), '..', 'share', 'gromacs', 'top'))

    return '/usr/local/gromacs/share/gromacs/top'

def interleave_dicts(dict0, dict1, dict2=None, dict3=None):
    if (dict2 is not None) and (dict3 is not None):
        if len(dict0) != len(dict1) or len(dict0) != len(dict2) or len(dict0) != len(dict3):
            raise ValueError('Dictionaries have unequal length')
            
        combined = {}
        for key0, key1, key2, key3 in zip(dict0, dict1, dict2, dict3):
            combined[key0] = dict0[key0]
            combined[key1] = dict1[key1]
            combined[key2] = dict2[key2]
            combined[key3] = dict3[key3]
    else:    
        if len(dict0) != len(dict1):
            raise ValueError('Dictionaries have unequal length')

        combined = {}
        for key0, key1 in zip(dict0, dict1):
            combined[key0] = dict0[key0]
            combined[key1] = dict1[key1]
        
    return combined


