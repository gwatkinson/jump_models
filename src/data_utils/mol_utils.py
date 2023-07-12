import datamol as dtm


def inchi_to_smiles(inchi: str) -> str:
    return dtm.to_smiles(dtm.from_inchi(inchi))
