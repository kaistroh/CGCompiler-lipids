"""Provides various functions related to GROMACS. NOTE: Functions are only tested for specific applications. """
#TODO cleanup

#from    math        import floor
#from    contextlib  import suppress
from    bisect      import bisect_left, bisect_right
#import  functools
import os

import  core.system  as system
#import  user.seq2itp_aminoacids as seq2itp

TOP_HEADER_SECTION          = "header"
TOP_SYSTEM_SECTION          = "system"
TOP_MOLECULES_SECTION       = "molecules"
ITP_MOLECULE_TYPE_SECTION   = "moleculetype"
ITP_ATOMS_SECTION           = "atoms"

class GROFileData:
    def __init__(self, header, atom_count, data, box_vector):
        self.header     = header
        self.atom_count = atom_count
        self.data       = data
        self.box_vector = box_vector

def TOP_get_system_charge(filepath_TOP, TOP_itp_directory_list=[]):
    def ITP_get_molecule_charges(filepath_ITP):
        class Molecule:
            def __init__(self, name):
                self.name = name
                self.charge = 0

        data = {}
        with open(filepath_ITP, 'rt') as file:
            state = None
            current_molecule = None
            for line in file:
                line = line.strip()
                if len(line) == 0 or line.startswith(";"):
                    continue

                if line.startswith("["):
                    if ITP_MOLECULE_TYPE_SECTION in line:
                        state = ITP_MOLECULE_TYPE_SECTION
                    elif ITP_ATOMS_SECTION in line:
                        state = ITP_ATOMS_SECTION
                    else:
                        state = None
                        if current_molecule is not None:
                            data[current_molecule.name] = current_molecule.charge
                    continue

                if state == ITP_MOLECULE_TYPE_SECTION:
                    current_molecule = Molecule(line.split()[0])
                elif state == ITP_ATOMS_SECTION:
                    current_molecule.charge += round(float(line.split()[6]))
        return data

    TOP_data = TOP_read(filepath_TOP)

    header_data = TOP_data[TOP_HEADER_SECTION]
    molecule_charge_data = {}
    TOP_itp_directory = ""
    for line in header_data:
        line = line.strip()
        if "#include" in line:
            itp_filename = line.split("\"")[1].split("\"")[0]
            if len(TOP_itp_directory_list) > 0:
                for dirname in TOP_itp_directory_list:
                    full_path = os.path.join(dirname, itp_filename)
                    if os.path.isfile(full_path):
                        TOP_itp_directory = dirname
                        break
            molecule_charge_data = {**ITP_get_molecule_charges(os.path.join(TOP_itp_directory, itp_filename)), **molecule_charge_data}

    total_charge_system = 0
    for line in TOP_data[TOP_MOLECULES_SECTION]:
        line = line.strip().split()
        if line[0] in molecule_charge_data:
            total_charge_system +=  molecule_charge_data[line[0]] * round(float(line[1]))
    return total_charge_system

def GRO_TOP_neutralize_system(filepath_input_GRO, filepath_input_TOP, filepath_output_GRO, filepath_output_TOP, name_Solvent, name_positive_ion, name_negative_ion, TOP_itp_directory_list=[]):
    system_charge = TOP_get_system_charge(filepath_input_TOP, TOP_itp_directory_list=TOP_itp_directory_list)
    data_TOP = TOP_read(filepath_input_TOP)
    data_GRO = GRO_read(filepath_input_GRO)

    if system_charge == 0:
        pass # No need to change the files
    else:
        #TOP#
        output_data_TOP = []
        molecules = data_TOP[TOP_MOLECULES_SECTION]
        for i in range(len(molecules)):
            molecule=molecules[i].strip().split()

            if molecule[0] == name_Solvent:
                if system_charge > 0:
                    name_Ion = name_negative_ion
                elif system_charge < 0:
                    name_Ion = name_positive_ion
                output_data_TOP.append("" + name_Ion + " " + str(abs(system_charge)) + "\n")

                new_count = int(molecule[1]) - abs(system_charge)
                output_data_TOP.append("" + name_Solvent + " " + str(new_count) + "\n")
            else:
                output_data_TOP.append(molecules[i])

        data_TOP[TOP_MOLECULES_SECTION] = output_data_TOP

        #GRO#
        count_ION = abs(system_charge)

        for i in range(len(data_GRO.data)):
            if count_ION <= 0:
                break

            if name_Solvent in data_GRO.data[i][5:10].strip():
                data_GRO.data[i] = data_GRO.data[i][0:5] + "%-5s%5s" % (name_Ion, name_Ion) + data_GRO.data[i][15:]
                count_ION -= 1
    
    with open(filepath_output_TOP, 'wt') as file:
        for line in data_TOP[TOP_HEADER_SECTION]:
            file.write(line)
        file.write("\n")
        file.write("[" + TOP_SYSTEM_SECTION + "]\n")
        for line in data_TOP[TOP_SYSTEM_SECTION]:
            file.write(line)
        file.write("\n")
        file.write("[" + TOP_MOLECULES_SECTION + "]\n")
        for line in data_TOP[TOP_MOLECULES_SECTION]:
            file.write(line)

    with open(filepath_output_GRO, 'wt') as file:
        for line in data_GRO.header:
            file.write(line)
        file.write(str(data_GRO.atom_count) + "\n")
        for line in data_GRO.data:
            file.write(line)
        file.write(" ".join(data_GRO.box_vector))

def GRO_read_box(filepath):
    """
    Reads the box vector from GRO file.

    Args:
        filepath (string): Path to GRO file.
    Returns:
        (list): List of strings describing the box vector [v1(x), v2(y), v3(z), v1(y), v1(z), v2(x), v2(z), v3(x), v3(y)].
    """
    box_string = None
    data = system.read_text_file(filepath)
    
    for line in reversed(data):
        line = line.strip()
        if not len(line) == 0:
            box_string = line.split()
            break

    return box_string

def GRO_read(filepath_GRO):
    """
    Args:
        filepath_GRO (string):  Path to GRO file.
    Returns:
        GROFileData object, with members:
            header (string):    Title string of the file.
            atom_count (int):   Number of atoms in file.
            data (list):        List of strings, each string describing an atom. Strings contain line endings.
            box_vector (list):  List of strings describing the box vector.
    """
    file_data = system.read_text_file(filepath_GRO, strip=False)

    header      = file_data[0]
    atom_count  = int(file_data[1])
    data        = [file_data[i] for i in range(2, 2 + atom_count)]
    box_vector  = file_data[2 + atom_count].strip().split()

    return GROFileData(header, atom_count, data, box_vector)

def GRO_merge(filepath_GRO_base, filepath_GRO_insert, filepath_GRO_output):
    """
    Combines two GRO files (base + insert) into one, keeping the header and box vector of filepath_GRO_base.
    """
    data_base   = system.read_text_file(filepath_GRO_base)
    data_insert = system.read_text_file(filepath_GRO_insert)
    
    # Strip box from data_base
    for i in reversed(range(0, len(data_base))):
        line = data_base[i].strip()
        if not len(line) == 0:
            del(data_base[i])
            break
        del(data_base[i])
    
    data_output     = data_base + data_insert[2:]
    data_output[1]  = str(len(data_output) - 3) + "\n"
    system.write_text_file(filepath_GRO_output, data_output, add_newline=False)

def TOP_read(filepath):
    """
    Returns:
        (dict): Each key represents a section of the TOP/ITP file (e.g. [ atoms ] becomes key "atoms").
                The corresponding value is a list of strings containing the lines belonging to that section.
    """
    data_sections = {}
    with open(filepath, 'rt') as file:
        state = TOP_HEADER_SECTION
        data_sections[state] = []
        for row in file:
            row_stripped = row.strip()
            if len(row_stripped) == 0 or row_stripped.startswith(";"):
                continue
            if row_stripped.startswith("["):
                state = row_stripped.split("[", 1)[-1].split("]")[0].strip()
                if state not in data_sections:
                    data_sections[state] = []
                continue

            data_sections[state].append(row)

    return data_sections

def TOP_merge(filepath_TOP_base, filepath_TOP_insert, filepath_TOP_output, sections_list, sections_list_append):
    """
    Combines two TOP files into one (base + insert).
    
    Args:
        sections_list (list): List of strings describing which sections of 'filepath_TOP_base' will be written to the output file.
        sections_list_append (list): List of strings describing which sections of 'filepath_TOP_insert' are appended 
                                        to the sections of 'filepath_TOP_base' in the output file.
                                        Only sections also present in 'sections_list' are included.
    """
    data_sections_base      = TOP_read(filepath_TOP_base)
    data_sections_insert    = TOP_read(filepath_TOP_insert)
    header_union = None

    if TOP_HEADER_SECTION in sections_list and TOP_HEADER_SECTION in sections_list_append:
        header_combined = data_sections_base[TOP_HEADER_SECTION] + data_sections_insert[TOP_HEADER_SECTION]
        for i in range(len(header_combined)):
            include_filename    =  header_combined[i].split("\"", 1)[-1].split("\"")[0]
            header_combined[i]  = "#include \"" + include_filename + "\"\n"
        header_union = list(set(header_combined)) # Removes duplicates. Removes original order of list as well.

        #Ensure headers are in order in which they appear in header_combined
        i_sorted = []
        for i in range(len(header_union)):
            i_sorted.append(header_combined.index(header_union[i]))
        
        i_sorted = sorted(i_sorted)
        for i in range(len(i_sorted)):
            header_union[i] = header_combined[i_sorted[i]]

    with open(filepath_TOP_output, 'wt') as file:
        for section in sections_list:
            if section == TOP_HEADER_SECTION and header_union is not None:
                for line in header_union:
                    file.write(line)
                continue
            elif not section == TOP_HEADER_SECTION:
                file.write("\n[" + section + "]\n")
            else:
                continue
            for row in data_sections_base[section]:
                file.write(row)
            if section in sections_list_append:
                for row in data_sections_insert[section]:
                    file.write(row)


def NDX_write(filepath_NDX, index_dict):
    """
    Args:
        filepath_NDX (string): Path to output NDX file.
        index_dict (dict(string, int)): Dictionary with keys describing the index group names 
                                        and values describing which atoms belong to that group.
    """
    with open(filepath_NDX, 'wt') as file:
        for group, indices in index_dict.items():
            count = 0
            file.write("[ " + group + " ]\n")
            for index in indices:
                file.write(str(index) + "\t")
                count+=1
                if count >= 15:
                    file.write("\n")
                    count = 0
            file.write("\n")

def NDX_read(filepath_NDX):
    """Returns index_dict, see NDX_write"""

    NDX_dict = {}
    state = None
    with open(filepath_NDX, 'rt') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("["):
                state = line.split("[")[1].split("]")[0].strip()
                if state not in NDX_dict:
                    NDX_dict[state] = []
                continue
            
            if state is None:
                continue
            else:
                indices = line.split()
                for index in indices:
                    NDX_dict[state].append(int(index))
                continue
    
    return NDX_dict

def NDX_merge(filepath_NDX_base, filepath_NDX_insert, filepath_NDX_output):
    """Merges two index files"""
    ndx_base = NDX_read(filepath_NDX_base)
    ndx_insert = NDX_read(filepath_NDX_insert)

    ndx_merged = {**ndx_base, **ndx_insert}

    NDX_write(filepath_NDX_output, ndx_merged)

def NDX_from_GRO(filepath_GRO, filepath_NDX_out, resnm_to_group_dict):
    """
    Produces an NDX file from a GRO file according to a reference dictionary.
    
    Args:
        resnm_to_group_dict (dict(string, string/tuple)): Dict with keys describing residue names, 
                                                    and values describing to which index group they belong.
    """
    data        = GRO_read(filepath_GRO).data
    NDX_dict    = {}

    for i in range(len(data)):
        resnm = data[i][5:10].strip()
        groups = resnm_to_group_dict[resnm]

        if not isinstance(groups, tuple):
            groups = (groups,)

        for group in groups:
            if group not in NDX_dict:
                NDX_dict[group] = []
            NDX_dict[group].append(str(i+1)) # atom index counts from 1 instead of 0

    NDX_write(filepath_NDX_out, NDX_dict)

def NDX_split_group(filepath_NDX_input, filepath_NDX_output, resname, atoms_per_residue, resname_split):
    """
    Split an index group into multiple smaller groups, keeping the original group intact.

    Args:
        resname (string): Index group to split
        atoms_per_residue (int): Number of indices that go into each new group
        resname_split (string): Name for the new groups, this name will be appended by an index number (e.g. "CL" -> "CL1" "CL2" "CL3" ...)
    """
    NDX_dict = NDX_read(filepath_NDX_input)

    if resname not in NDX_dict:
        raise KeyError("Error: resname (" + resname + ") not found in index file (" + filepath_NDX_input + ")")

    indices = NDX_dict[resname]

    count = 0
    index_new_residue = 1
    new_residue = []
    for index in indices:
        new_residue.append(index)
        count += 1

        if count >= atoms_per_residue:
            NDX_dict[resname_split + str(index_new_residue)] = new_residue

            index_new_residue += 1
            count = 0
            new_residue = []
    
    NDX_write(filepath_NDX_output, NDX_dict)

def GMX_average_ENERGY_output(filepath_GMX_ENERGY_output):
    # Returns dict containing energy selections and average

    data_dict = {}
    output_column_template = ["Energy", "Average"]

    with open(filepath_GMX_ENERGY_output, 'rt') as file:
        state = False
        for line in file:
            line = line.strip()

            if state:
                if line.startswith('-'):
                    continue
                columns = line.split()
                data_dict[columns[0]] = float(columns[1])
                continue

            if line.startswith("Energy"):
                columns = line.split()
                if columns[0:2] == output_column_template:
                    state = True
                    continue
    
    return data_dict

def XVG_read(filepath_XVG):
    """Returns a dictionary containing:

        "x":        list of x data points

        "y":        list of y data point lists,
                    first index indicates data column

        "title":    string

        "legend":   list of strings matching the data columns in the file

        "xlabel":   string

        "ylabel":   string
    """
    XVG_dict    = {}

    x           = []
    ylist       = None
    legend      = []
    title       = ""
    xlabel      = ""
    ylabel      = ""

    with open(filepath_XVG, "rt") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                splt = line.split()
                if splt[1] == "title":
                    title = " ".join(splt[2:])
                elif splt[1] == "xaxis":
                    xlabel = " ".join(splt[3:])
                elif splt[1] == "yaxis":
                    ylabel = " ".join(splt[3:])
                elif len(splt) > 3 and splt[2] == "legend":
                    legend.append(" ".join(splt[3:]))
                continue

            data = line.split()
            if ylist is None:
                ylist = [[] for i in range(len(data)-1)]

            x.append(float(data[0]))
            data = data[1:]
            for i, value in enumerate(data):
                ylist[i].append(float(value))

    XVG_dict["x"]       = x
    XVG_dict["y"]       = ylist
    XVG_dict["title"]   = title
    XVG_dict["legend"]  = legend
    XVG_dict["xlabel"]  = xlabel
    XVG_dict["ylabel"]  = ylabel

    return XVG_dict

def XVG_ymax(filepath_XVG, x_range=None, column_index=0):
    XVG_dict = XVG_read(filepath_XVG)
    x_data = XVG_dict["x"]
    y_data = XVG_dict["y"][column_index]

    if x_range:
        idx_range = (bisect_left(x_data, x_range[0]), bisect_right(x_data, x_range[1]))
        y_data = y_data[idx_range[0]:idx_range[1]]

    ymax = y_data[0]
    for value in y_data:
        if value > ymax:
            ymax = value

    return ymax

def editMDP_Tm_biphasic_production(filepath_in, filepath_out, T, nsteps):
    baseMDP = open(filepath_in, 'r')
    baseMDPLines = baseMDP.readlines()

    file = open(filepath_out, 'w')
   
    for line in baseMDPLines:
        if 'nsteps ' in line:
            newline = 'nsteps                   = %d\n' %nsteps
            file.write(newline)
        elif 'ref_t ' in line:
            newline = 'ref_t                    = %.2f %.2f\n' %(T, T)
            file.write(newline)
        elif 'gen_temp' in line:
            newline = 'gen_temp                  = %.2f\n' %(T)
            file.write(newline)
        else:
            file.write(line)
            
    file.close()

def editMDP_Tm_biphasic_equil(filepath_in, filepath_out, T_solv, T_fluid, T_gel, nsteps):
    baseMDP = open(filepath_in, 'r')
    baseMDPLines = baseMDP.readlines()

    file = open(filepath_out, 'w')
    
    for line in baseMDPLines:
        if 'nsteps ' in line:
            newline = 'nsteps                   = %d\n' %nsteps
            file.write(newline)
        elif 'ref_t ' in line:
            newline = 'ref_t                    = %.2f %.2f %.2f\n' %(T_solv, T_fluid, T_gel)
            file.write(newline)
        elif 'gen_temp' in line:
            newline = 'gen_temp                  = %.2f\n' %(T_gel)
            file.write(newline)
        else:
            file.write(line)
            
    file.close()

