"""
trait_mapper.py

Maps structured binary trait output
to lookup-compatible trait names.
"""


class TraitMapper:
    """
    Converts binary trait dictionary
    into list of lookup-compatible trait names.
    """

    def __init__(self):

        # Mapping between pipeline output keys
        # and lookup table attribute names
        self.mapping = {
            "wide_set": "Wide Set",
            "big_nose": "Large Nose",
            "broad_face": "Broad Face",
            "high_cheekbones": "High Cheekbones",
            "double_chin": "Double Chin"
        }

    def map(self, binary_traits: dict):
        """
        binary_traits example:
        {
            "wide_set": True,
            "big_nose": False,
            "broad_face": True,
            "high_cheekbones": False,
            "double_chin": True
        }

        Returns:
            ["Wide Set", "Broad Face", "Double Chin"]
        """

        detected_traits = []

        for key, value in binary_traits.items():

            if key in self.mapping and value:
                detected_traits.append(self.mapping[key])

        return detected_traits