from src.modules.compound_transforms.base_compound_transform import DefaultCompoundTransform


class COATITransform(DefaultCompoundTransform):
    """Pass the string through the COATI model."""

    def mol_to_feat(self, mol: str):
        return mol
