from typing import Any, Optional, Union, Tuple

import yaml

from sleap.util import get_config_file
from sleap.instance import Instance, Track, Node
from sleap.io.dataset import Labels


class ColorManager(object):
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

        self.distinctly_color = "instances"
        self.color_predicted = True

        self.index_mode = "cycle"
        self._index_mode_functions = dict(
            cycle=lambda i, c: i % c, clip=lambda i, c: min(i, c - 1)
        )

        self.set_palette(palette)

        self.uncolored_prediction_color = (250, 250, 10)
        self.default_pen_width = 1
        self.medium_pen_width = 1.5
        self.thick_pen_width = 3

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
    def palette(self, palette):
        self._palette = palette

        if isinstance(palette, str):
            self.index_mode = "clip" if palette.endswith("+") else "cycle"

            if palette in self._palettes:
                self._color_map = self._palettes[palette]
            else:
                self._color_map = self._palettes["standard"]
        else:
            self._color_map = palette

    @property
    def palette_names(self):
        """Gets list of palette names."""
        return self._palettes.keys()

    @property
    def tracks(self):
        """Gets tracks for project."""
        if self.labels:
            return self.labels.tracks
        return []

    def set_palette(self, palette):
        """Functional alias for palette property setter."""
        self.palette = palette

    def fix_index(self, idx):
        """Returns an index within range of color palette."""
        return self._index_mode_functions[self.index_mode](idx, len(self._color_map))

    def get_color_by_idx(self, idx):
        """Returns color tuple corresponding to item index."""
        color_idx = self.fix_index(idx)
        try:
            return tuple(map(int, self._color_map[color_idx].split(",")))
        except:
            raise ValueError(f"Invalid color: {self._color_map[color_idx]}")

    def get_pseudo_track_index(self, instance: "Instance") -> Union[Track, int]:
        """
        Returns an index for giving track colors to instances without track.
        """
        if instance.track:
            return instance.track
        if not instance.frame:
            return 0

        non_track_instances = [
            inst for inst in instance.frame.instances_to_show if inst.track is None
        ]

        return len(self.tracks) + non_track_instances.index(instance)

    def get_track_color(self, track: Union[Track, int]) -> Tuple[int, int, int]:
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

    def get_item_color(
        self,
        item: Any,
        parent_instance: Optional[Instance] = None,
        parent_skeleton: Optional["Skeleton"] = None,
    ) -> Tuple[int, int, int]:
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
