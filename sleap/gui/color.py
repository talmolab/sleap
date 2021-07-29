"""
Logic for determining what color/width to draw instance nodes/edges.

The color can be determined by the current color palette as well as settings
on the `ColorManager` object:

* distinctly_color: "instances", "nodes", or "edges"
* color_predicted: whether to use colors for predicted instances, or just plot
  them in yellow/grey

Initial color palette (and other settings, like default line width) is read
from user preferences but can be changed after object is created.
"""
from typing import Any, Iterable, Optional, Union, Text, Tuple

import yaml

from sleap.util import get_config_file
from sleap.instance import Instance, Track, Node
from sleap.io.dataset import Labels
from sleap.prefs import prefs


ColorTupleStringType = Text
ColorTupleType = Tuple[int, int, int]


class ColorManager:
    """Class to determine color to use for track.

    The color depends on the order of the tracks in `Labels` object,
    so we need to initialize with `Labels`.

    Args:
        labels: The :class:`Labels` dataset which contains the tracks for
            which we want colors.
        palette: String with the color palette name to use.
    """

    def __init__(self, labels: Labels = None, palette: str = "standard"):
        self.labels = labels

        with open(get_config_file("colors.yaml"), "r") as f:
            self._palettes = yaml.load(f, Loader=yaml.SafeLoader)

        self._color_map = []
        self.distinctly_color = "instances"
        self.color_predicted = True

        self.index_mode = "cycle"
        self._index_mode_functions = dict(
            cycle=lambda i, c: i % c, clip=lambda i, c: min(i, c - 1)
        )

        self.set_palette(palette)

        self.uncolored_prediction_color = (250, 250, 10)

        if prefs["bold lines"]:
            self.thick_pen_width = 6
        else:
            self.thick_pen_width = 3

        self.medium_pen_width = self.thick_pen_width // 2
        self.default_pen_width = max(1, self.thick_pen_width // 4)

    @property
    def labels(self):
        """Gets or sets labels dataset for which we are coloring tracks."""
        return self._labels

    @labels.setter
    def labels(self, val):
        self._labels = val

    @property
    def palette(self):
        """Gets or sets palette (by name)."""
        return self._palette

    @palette.setter
    def palette(self, palette: Union[Text, Iterable[ColorTupleStringType]]):
        self._palette = palette

        if isinstance(palette, Text):
            self.index_mode = "clip" if palette.endswith("+") else "cycle"

            if palette in self._palettes:
                self._color_map = self._palettes[palette]
            else:
                # Can't find palette by name so just use standard palette.
                self._color_map = self._palettes["standard"]
        else:
            # If palette is not given by name, it should be list of
            # "r,g,b" strings.
            self._color_map = palette

    @property
    def palette_names(self) -> Iterable[Text]:
        """Gets list of palette names."""
        return self._palettes.keys()

    @property
    def tracks(self) -> Iterable[Track]:
        """Gets tracks for project."""
        if self.labels:
            return self.labels.tracks
        return []

    def set_palette(self, palette: Union[Text, Iterable[ColorTupleStringType]]):
        """Functional alias for palette property setter."""
        self.palette = palette

    def fix_index(self, idx: int) -> int:
        """Returns an index within range of color palette."""
        return self._index_mode_functions[self.index_mode](idx, len(self._color_map))

    def get_color_by_idx(self, idx: int) -> ColorTupleType:
        """Returns color tuple corresponding to item index."""
        color_idx = self.fix_index(idx)
        return self.color_to_tuple(self._color_map[color_idx])

    @staticmethod
    def color_to_tuple(color: Union[Text, Iterable[int]]) -> ColorTupleType:
        """Convert and ensure color is (r, g, b)-tuple."""
        if isinstance(color, Text):
            split_string = color.split(",")
            if len(split_string) != 3:
                raise ValueError(f"Color '{color}' is not 'r,g,b' string.")
            try:
                result = tuple(map(int, split_string))
                return result
            except:
                raise ValueError(f"Color '{color}' is not 'r,g,b' string.")

        if len(color) != 3:
            raise ValueError(f"Color '{color}' is not (r,g,b) tuple.")

        try:
            result = tuple(map(int, color))
            return result
        except:
            raise ValueError(f"Color '{color}' is not (r,g,b) tuple.")

    def get_pseudo_track_index(self, instance: "Instance") -> Union[Track, int]:
        """
        Returns an index for giving track colors to instances without track.
        """
        if instance.track:
            return instance.track
        if not instance.frame:
            return 0

        untracked_user_instances = [
            inst for inst in instance.frame.user_instances if inst.track is None
        ]
        untracked_predicted_instances = [
            inst for inst in instance.frame.predicted_instances if inst.track is None
        ]

        return len(self.tracks) + (
            untracked_user_instances + untracked_predicted_instances
        ).index(instance)

    def get_track_color(self, track: Union[Track, int]) -> ColorTupleType:
        """Returns the color to use for a given track.

        Args:
            track: `Track` object or an int
        Returns:
            (r, g, b)-tuple
        """
        track_idx = self.tracks.index(track) if isinstance(track, Track) else track
        if track_idx is None:
            return (0, 0, 0)

        return self.get_color_by_idx(track_idx)

    @classmethod
    def is_sequence(cls, item) -> bool:
        """Returns whether item is a tuple or list."""
        return isinstance(item, tuple) or isinstance(item, list)

    @classmethod
    def is_edge(cls, item) -> bool:
        """Returns whether item is an edge, i.e., pair of nodes."""
        return cls.is_sequence(item) and len(item) == 2 and cls.is_node(item[0])

    @staticmethod
    def is_node(item) -> bool:
        """Returns whether item is a node, i.e., Node or node name."""
        return isinstance(item, Node) or isinstance(item, str)

    @staticmethod
    def is_predicted(instance) -> bool:
        """Returns whether instance is predicted."""
        return hasattr(instance, "score")

    def get_item_pen_width(
        self, item: Any, parent_instance: Optional[Instance] = None
    ) -> float:
        """Gets width of pen to use for drawing item."""

        if self.is_node(item):
            if self.distinctly_color == "nodes":
                return self.thick_pen_width

            if self.is_predicted(parent_instance):

                is_first_node = item == parent_instance.skeleton.nodes[0]
                return self.thick_pen_width if is_first_node else self.medium_pen_width
            else:
                return self.medium_pen_width

        if self.is_edge(item):
            if self.distinctly_color == "edges":
                return self.thick_pen_width

        return self.default_pen_width

    def get_item_type_pen_width(self, item_type: str) -> float:
        """Gets pen width to use for given item type (as string)."""
        if item_type == "node":
            if self.distinctly_color == "nodes":
                return self.thick_pen_width
            return self.medium_pen_width

        if item_type == "edge":
            if self.distinctly_color == "edges":
                return self.thick_pen_width

        return self.default_pen_width

    def get_item_color(
        self,
        item: Any,
        parent_instance: Optional[Instance] = None,
        parent_skeleton: Optional["Skeleton"] = None,
    ) -> ColorTupleType:
        """Gets (r, g, b) tuple of color to use for drawing item."""

        if not parent_instance and isinstance(item, Instance):
            parent_instance = item

        if not parent_skeleton and hasattr(parent_instance, "skeleton"):
            parent_skeleton = parent_instance.skeleton

        is_predicted = False
        if parent_instance and self.is_predicted(parent_instance):
            is_predicted = True

        if is_predicted and not self.color_predicted:
            if isinstance(item, Node):
                return self.uncolored_prediction_color

            return (128, 128, 128)

        if self.distinctly_color == "instances" or hasattr(item, "track"):
            track = None
            if hasattr(item, "track"):
                track = item.track
            elif parent_instance:
                track = parent_instance.track

            if track is None and parent_instance:
                # Get an index for items without track
                track = self.get_pseudo_track_index(parent_instance)

            return self.get_track_color(track=track)

        if self.distinctly_color == "nodes" and parent_skeleton:
            node = None
            if isinstance(item, Node):
                node = item
            elif self.is_edge(item):
                # use dst node for coloring edge
                node = item[1]

            if node:
                node_idx = parent_skeleton.node_to_index(node)
                return self.get_color_by_idx(node_idx)

            # return (255, 0, 0)

        if self.distinctly_color == "edges" and parent_skeleton:
            edge_idx = 0
            if self.is_edge(item):
                edge_idx = parent_skeleton.edge_to_index(*item)
            elif self.is_node(item):
                for i, (src, dst) in enumerate(parent_skeleton.edges):
                    if dst == item:
                        edge_idx = i
                        break

            return self.get_color_by_idx(edge_idx)

        return (0, 0, 0)
