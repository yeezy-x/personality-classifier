"""
personality_engine.py

Weighted lookup-based personality inference.
"""

class PersonalityEngine:

    def __init__(self, lookup):
        self.lookup = lookup

    def infer(self, feature_list, top_k=3):
        """
        feature_list : list[str]

        Returns structured personality output.
        """

        if not feature_list:
            return {"error": "No traits detected"}

        scores = self.lookup.score_traits(feature_list)

        if scores is None or scores.empty:
            return {"error": "No matching traits in lookup"}

        # Sort descending (highest influence first)
        scores_sorted = scores.sort_values(ascending=False)

        top_traits = scores_sorted.head(top_k)
        lowest_traits = scores_sorted.tail(top_k)

        return {
            "detected_traits": feature_list,
            "top_traits": top_traits.to_dict(),
            "lowest_traits": lowest_traits.to_dict()
        }