"""
# removing obsolete models including git history:
https://help.github.com/en/articles/removing-files-from-a-repositorys-history

# model persistence
https://cmry.github.io/notes/serialize
https://stackabuse.com/scikit-learn-save-and-restore-models/
https://github.com/scikit-learn/scikit-learn/issues/10319
https://stackoverflow.com/questions/20156951/how-do-i-find-which-attributes-my-tree-splits-on-when-using-scikit-learn
http://thiagomarzagao.com/2015/12/08/saving-TfidfVectorizer-without-pickles/

#############################################
#Forest to trees Extraction code:
import joblib
import glob

for model in glob.glob("*.model"):
        print(model)
        forest = joblib.load(model).estimators_
        for idx, tree in enumerate(forest):
                joblib.dump(tree, "./epsilon_4/{}_{}.model".format(model.split(".")[0], idx), compress = 9)
#############################################
"""

def validate_models(model_dict, epsilon):
    import numpy as np
    from rdkit import Chem
    import tqdm
    try:
        smiles = np.load(_get_data_filename("validated_results/test_smiles.npy"), allow_pickle = True)
        charges =np.load(_get_data_filename("validated_results/test_charges_{}.npy".format(epsilon)), allow_pickle = True)
    except ValueError:
        raise ValueError("No model for epsilon value of {}".format(epsilon))

    print("Checking through molecule dataset, stops when discrepencies are observed...")
    for s,c in tqdm.tqdm(list(zip(smiles, charges))):
        if np.any(~np.isclose(get_charges(Chem.AddHs(Chem.MolFromSmiles(s)), model_dict) , c, atol = 0.01)):
            print("No close match for {}, validation terminated.".format(smiles))
            return



def _get_data_filename(relative_path):
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
    # from sklearn.externals import joblib
    import joblib
    # supported elements (atomic numbers)
    # H, C, N, O, F, P, S, Cl, Br, I
    element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    elementdict = {8:"O", 7:"N", 6:"C", 1:"H", \
                   9:"F", 15:"P", 16:"S", 17:"Cl", \
                   35:"Br", 53:"I"}
    #directory, containing the models
    progress_bar = True
    try:
        import tqdm
    except ImportError:
        progress_bar = False
    print("Loading models...")
    try:
        if progress_bar:
            rf = {element : [joblib.load(_get_data_filename("epsilon_{}/{}_{}.model".format(epsilon, elementdict[element], i))) for i in range(100)] for element in tqdm.tqdm(element_list)}

        else:
            rf = {element : [joblib.load(_get_data_filename("epsilon_{}/{}_{}.model".format(epsilon, elementdict[element], i))) for i in range(100)] for element in element_list}

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
    from rdkit import DataStructs, Chem
    from rdkit.Chem import AllChem
    import numpy as np

    num_atoms = mol.GetNumAtoms()
    if num_atoms != Chem.AddHs(mol).GetNumAtoms():
        import warnings
        warnings.warn("Have you added hydrogens to the molecule?", UserWarning)

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

def add_charges_to_mol(mol, model_dict = None,  charges = None, property_name = "PartialCharge"):
    """
    if charges is None, perform fitting using `get_charges`, for this `model_dict` needs to be provided
    """
    if type(charges) is list:
        assert mol.GetNumAtoms() == len(charges)
    elif charges is None and model_dict is not None:
        charges = get_charges(mol, model_dict)
    for i,atm in enumerate(mol.GetAtoms()):
        atm.SetDoubleProp(property_name, charges[i])
    return mol


def _draw_mol_with_property( mol, property, **kwargs ):
    """
    http://rdkit.blogspot.com/2015/02/new-drawing-code.html

    Parameters
    ---------
    property : dict
        key atom idx, val the property (need to be stringfiable)
    """
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem

    def run_from_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False


    AllChem.Compute2DCoords(mol)
    for idx in property:
        # opts.atomLabels[idx] =
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', "({})".format( str(property[idx])))

    mol = Draw.PrepareMolForDrawing(mol, kekulize=False) #enable adding stereochem

    if run_from_ipython():
        from IPython.display import SVG, display
        if "width" in kwargs and type(kwargs["width"]) is int and "height" in kwargs and type(kwargs["height"]) is int:
            drawer = Draw.MolDraw2DSVG(kwargs["width"], kwargs["height"])
        else:
            drawer = Draw.MolDraw2DSVG(500,250)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        display(SVG(drawer.GetDrawingText().replace("svg:", "")))
    else:
        if "width" in kwargs and type(kwargs["width"]) is int and "height" in kwargs and type(kwargs["height"]) is int:
            drawer = Draw.MolDraw2DCairo(kwargs["width"], kwargs["height"])
        else:
            drawer = Draw.MolDraw2DCairo(500,250) #cairo requires anaconda rdkit
        # opts = drawer.drawOptions()
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        #
        # with open("/home/shuwang/sandbox/tmp.png","wb") as f:
        #     f.write(drawer.GetDrawingText())

        import io
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        buff = io.BytesIO()
        buff.write(drawer.GetDrawingText())
        buff.seek(0)
        plt.figure()
        i = mpimg.imread(buff)
        plt.imshow(i)
        plt.show()
        # display(SVG(drawer.GetDrawingText()))


def visualise_charges(mol, show_hydrogens = False, property_name = "PartialCharge" , **kwargs):
    if not show_hydrogens:
        from rdkit import Chem
        mol = Chem.RemoveHs(mol)

    atom_mapping = {}
    for idx,atm in enumerate(mol.GetAtoms()):
        try:
            #currently only designed with partial charge in mind
            # keeps 2.s.f, and remove starting `0.``
            tmp =  atm.GetProp(property_name)
            try:
                tmp =  str("{0:.2g}".format(float(tmp)))
                if tmp[0:3] == "-0.":
                    tmp = "-" + tmp[2:]
                elif tmp[0:2] == "0.":
                    tmp = tmp[1:]
            except:
                pass
            atom_mapping[idx] = tmp
        except Exception as e:
            print("Failed at atom number {} due to {}".format(idx, e))
            return
    _draw_mol_with_property(mol, atom_mapping, **kwargs)


def visualize_charges(mol, show_hydrogens = False, property_name = "PartialCharge", **kwargs ):
    return visualise_charges(mol, show_hydrogens, property_name, **kwargs)
