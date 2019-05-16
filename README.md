# mlddec
Random forest models for predicting atomic partial charges, fitted to DFT + DDEC6 calculations.

# usage
```python
import mlddec

epsilon = 4
models  = mlddec.load_models(epsilon)
charges = mlddec.get_charges(mol, models)
mlddec.add_charges_to_mol(mol, charges)

```
