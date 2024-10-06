from typing import Optional

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcNumRotatableBonds

from skfp.bases.base_filter import BaseFilter


class TiceHerbicidesFilter(BaseFilter):
    r"""
    Tice rule for herbicides.

    Rule established based on statistical analysis of herbicide molecules [1]_.
    Designed specifically for herbicides, not general pesticides or other agrochemicals.

    Molecule must fulfill conditions:

        - 150 <= molecular weight <= 500
        - logP <= 3.5
        - HBD <= 3
        - 2 <= HBA <= 12
        - number of rotatable bonds <= 11

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when generating conformers.

    References
    -----------
    .. [1] `Tice, C.M.
        "Selecting the right compounds for screening:
        does Lipinski's Rule of 5 for pharmaceuticals apply to agrochemicals?"
        Pest. Manag. Sci., 57: 3-16.
        <https://doi.org/10.1002/1526-4998(200101)57:1\<3::AID-PS269\>3.0.CO;2-6>`_

    Examples
    ----------
    >>> from skfp.preprocessing import TiceHerbicidesFilter
    >>> smiles = ["OCCNc1nc2ccc(Cl)cc2[nH]1", "Nc1nnc(-c2ccco2)s1", "S=C1NCCS1"]
    >>> filt = TiceHerbicidesFilter()
    >>> filt
    TiceHerbicidesFilter()
    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['OCCNc1nc2ccc(Cl)cc2[nH]1', 'Nc1nnc(-c2ccco2)s1']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            allow_one_violation=allow_one_violation,
            return_indicators=return_indicators,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _apply_mol_filter(self, mol: Mol) -> bool:
        rules = [
            150 <= MolWt(mol) <= 500,
            MolLogP(mol) <= 3.5,
            CalcNumHBD(mol) <= 3,
            2 <= CalcNumHBA(mol) <= 12,
            CalcNumRotatableBonds(mol) <= 11,
        ]
        passed_rules = sum(rules)

        if self.allow_one_violation:
            return passed_rules >= len(rules) - 1
        else:
            return passed_rules == len(rules)