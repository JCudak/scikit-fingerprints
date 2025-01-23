from skfp.datasets.moleculenet import load_moleculenet_benchmark
from skfp.filters import *
from skfp.fingerprints import *

datasets = load_moleculenet_benchmark(subset="classification")
forbidden_datasets = ["MUV", "Tox21", "ToxCast", "PCBA"]

filter_dict = {
    "No Filter": None,
    "Lipinski": LipinskiFilter(),
    "BeyondRo5": BeyondRo5Filter(),
    "BMS": BMSFilter(),
    "Brenk": BrenkFilter(),
    "faf4_druglike": FAF4DruglikeFilter(),
    "faf4_leadlike": FAF4LeadlikeFilter(),
    "ghose": GhoseFilter(),
    "hao": HaoFilter(),
    "inpharmatica": InpharmaticaFilter(),
    "lint": LINTFilter(),
    "mlsmr": MLSMRFilter(),
    "mol_weight": MolecularWeightFilter(),
    "nibr": NIBRFilter(),
    "nih": NIHFilter(),
    "oprea": OpreaFilter(),
    "pains": PAINSFilter(),
    "pfizer": PfizerFilter(),
    # "reos": REOSFilter(),
    # "rule_of_2" : RuleOfTwoFilter(),
    "rule_of_3": RuleOfThreeFilter(),
    "rule_of_4": RuleOfFourFilter(),
    "rule_of_xu": RuleOfXuFilter(),
    "surechembl": SureChEMBLFilter(),
    "tice_herebicides": TiceHerbicidesFilter(),
    "tice_insecticides": TiceInsecticidesFilter(),
    "valence_discovery": ValenceDiscoveryFilter(),
    "zinc_basic": ZINCBasicFilter(),
    "zinc_druglike": ZINCDruglikeFilter(),
}

fingerprint_classes = {
    "AtomPairFingerprint": AtomPairFingerprint,
    "AutocorrFingerprint": AutocorrFingerprint,
    "AvalonFingerprint": AvalonFingerprint,
    # "E3FPFingerprint": E3FPFingerprint, OUT
    "ECFPFingerprint": ECFPFingerprint,
    # "ElectroShapeFingerprint": ElectroShapeFingerprint, OUT
    "ERGFingerprint": ERGFingerprint,
    "EStateFingerprint": EStateFingerprint,
    "FunctionalGroupsFingerprint": FunctionalGroupsFingerprint,
    # "GETAWAYFingerprint": GETAWAYFingerprint, OUT
    "GhoseCrippenFingerprint": GhoseCrippenFingerprint,
    "KlekotaRothFingerprint": KlekotaRothFingerprint,
    "LaggnerFingerprint": LaggnerFingerprint,
    "LayeredFingerprint": LayeredFingerprint,
    "LingoFingerprint": LingoFingerprint,
    "MACCSFingerprint": MACCSFingerprint,
    "MAPFingerprint": MAPFingerprint,
    "MHFPFingerprint": MHFPFingerprint,
    "MordredFingerprint": MordredFingerprint,
    # "MORSEFingerprint": MORSEFingerprint, OUT
    "MQNsFingerprint": MQNsFingerprint,
    "PatternFingerprint": PatternFingerprint,
    "PharmacophoreFingerprint": PharmacophoreFingerprint,
    "PhysiochemicalPropertiesFingerprint": PhysiochemicalPropertiesFingerprint,
    "PubChemFingerprint": PubChemFingerprint,
    # "RDFFingerprint": RDFFingerprint, OUT
    "RDKit2DDescriptorsFingerprint": RDKit2DDescriptorsFingerprint,
    "RDKitFingerprint": RDKitFingerprint,
    "SECFPFingerprint": SECFPFingerprint,
    "TopologicalTorsionFingerprint": TopologicalTorsionFingerprint,
    # "USRFingerprint": USRFingerprint, OUT
    # "USRCATFingerprint": USRCATFingerprint, OUT
    "VSAFingerprint": VSAFingerprint,
    # "WHIMFingerprint": WHIMFingerprint OUT
}
