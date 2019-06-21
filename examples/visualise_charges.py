from rdkit import Chem
import mlddec


epsilon = 4
models  = mlddec.load_models(epsilon)

mol = Chem.MolFromSmiles("N[C@@H](C)CCCC(=O)")
mol = Chem.AddHs(mol)

mlddec.add_charges_to_mol(mol, models)
mlddec.visualize_charges(mol, show_hydrogens = False)
