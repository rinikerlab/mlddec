"""
model persistence
https://cmry.github.io/notes/serialize
https://stackabuse.com/scikit-learn-save-and-restore-models/
https://github.com/scikit-learn/scikit-learn/issues/10319
https://stackoverflow.com/questions/20156951/how-do-i-find-which-attributes-my-tree-splits-on-when-using-scikit-learn
http://thiagomarzagao.com/2015/12/08/saving-TfidfVectorizer-without-pickles/

"""
def get_data_filename(relative_path):
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``openforcefield/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).
    """
    import os
    from pkg_resources import resource_filename
    fn = resource_filename('mlddec', os.path.join('data', relative_path))
    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)
    return fn

def load_models(epsilon = 4):
    from sklearn.externals import joblib
    # supported elements (atomic numbers)
    # H, C, N, O, F, P, S, Cl, Br, I
    element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    elementdict = {8:"O", 7:"N", 6:"C", 1:"H", \
                   9:"F", 15:"P", 16:"S", 17:"Cl", \
                   35:"Br", 53:"I"}
    #directory, containing the models
    try:
        rf = {element : [joblib.load(get_data_filename("epsilon_{}/{}_{}.model".format(epsilon, elementdict[element], i))) for i in range(100)] for element in element_list}
    except ValueError:
        raise ValueError("No model for epsilon value of {}".format(epsilon))
    return rf

def get_charges(mol, model_dict):
    """
    Parameters
    -----------
    mol : rdkit molecule
    model_dict : dictionary of random forest models
    """
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    import numpy as np
    num_atoms = mol.GetNumAtoms()
    element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

    # maximum path length in atompairs-fingerprint
    APLength = 4

    # check for unknown elements
    curr_element_list = []
    for at in mol.GetAtoms():
      element = at.GetAtomicNum()
      if element not in element_list:
          raise ValueError("Error: element {} has not been parameterised".format(element))
      curr_element_list.append(element)
    curr_element_list = set(curr_element_list)

    pred_q = [0]*num_atoms
    sd_rf = [0]*num_atoms
    # loop over the atoms
    for i in range(num_atoms):
      # generate atom-centered AP fingerprint
      fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, maxLength=APLength, fromAtoms=[i])
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(fp, arr)
      # get the prediction by each tree in the forest
      element = mol.GetAtomWithIdx(i).GetAtomicNum()
      per_tree_pred = [tree.predict(arr.reshape(1,-1)) for tree in model_dict[element]]
      # then average to get final predicted charge
      pred_q[i] = np.average(per_tree_pred)
      # and get the standard deviation, which will be used for correction
      sd_rf[i] = np.std(per_tree_pred)

    #########################
    # CORRECT EXCESS CHARGE #
    #########################

    # calculate excess charge
    deltaQ = sum(pred_q)- float(AllChem.GetFormalCharge(mol))
    charge_abs = 0.0
    for i in range(num_atoms):
      charge_abs += sd_rf[i] * abs(pred_q[i])
    deltaQ /= charge_abs
    # correct the partial charges

    return [(pred_q[i] - abs(pred_q[i]) * sd_rf[i] * deltaQ) for i in range(num_atoms)]

def add_charges_to_mol(mol, charges, property_name = "PartialCharge"):
    assert mol.GetNumAtoms() == len(charges)
    for i,atm in enumerate(mol.GetAtoms()):
        atm.SetDoubleProp(property_name, charges[i])
    # return mol
