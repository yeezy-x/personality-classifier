import pandas as pd


class PersonalityLookup:
    """
    Weighted lookup-based personality scoring system.
    """

    def __init__(self, lookup_path):

        self.df = pd.read_csv(lookup_path)

        # Fill missing weights with 0
        self.df = self.df.fillna(0)

        # Supported traits
        self.SUPPORTED_FEATURES = [
            "Wide Set",
            "Close Set",
            "Large Nose",
            "Long Nose",
            "Double Chin",
            "Broad Face",
            "High Cheekbones"
        ]

        # Keep only supported traits
        self.df = self.df[
            self.df["Attributes"].isin(self.SUPPORTED_FEATURES)
        ]

    def score_traits(self, feature_list):
        """
        feature_list : list[str]

        Returns:
            pandas Series of personality scores
        """

        if not feature_list:
            return None

        selected = self.df[
            self.df["Attributes"].isin(feature_list)
        ]

        if selected.empty:
            return None

        # Remove attribute column
        trait_matrix = selected.drop(columns=["Attributes"])

        # Sum weighted values
        scores = trait_matrix.sum()

        # Normalize by actual number of matched rows
        scores = scores / len(selected)

        return scores