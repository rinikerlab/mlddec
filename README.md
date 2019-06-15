# mlddec
Random forest models for predicting atomic partial charges, fitted to DFT + DDEC6 calculations.

Modified code based on [publication](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00663).

# usage
```python
import mlddec

epsilon = 4
models  = mlddec.load_models(epsilon)
charges = mlddec.get_charges(mol, models)
mlddec.add_charges_to_mol(mol, charges)

```


## Maintainer
Shuzhe Wang

