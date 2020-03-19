"""Utilities for config building and validation."""


def oneof(attrs_cls, must_be_set: bool = False):
    """Ensure that the decorated attrs class only has a single attribute set.

    This decorator is inspired by the `oneof` protobuffer field behavior.

    Args:
        attrs_cls: An attrs decorated class.
        must_be_set: If True, raise an error if none of the attributes are set. If not,
            error will only be raised if more than one attribute is set.

    Returns:
        The `attrs_cls` with an `__init__` method that checks for the number of
        attributes that are set.
    """
    # Check if the class is an attrs class at all.
    if not hasattr(attrs_cls, "__attrs_attrs__"):
        raise ValueError("Classes decorated with oneof must also be attr.s decorated.")

    # Pull out attrs generated class attributes.
    attribs = attrs_cls.__attrs_attrs__
    init_fn = attrs_cls.__init__

    # Define a new __init__ function that wraps the attrs generated one.
    def new_init_fn(self, *args, **kwargs):
        # Execute the standard attrs-generated __init__.
        init_fn(self, *args, **kwargs)

        # Check for attribs with set values.
        attribs_with_value = [
            attrib for attrib in attribs if getattr(self, attrib.name) is not None
        ]

        if len(attribs_with_value) > 1:
            # Raise error if more than one attribute is set.
            raise ValueError("Only one attribute of this class can be set (not None).")

        if len(attribs_with_value) == 0 and must_be_set:
            # Raise error if none are set.
            raise ValueError("At least one attribute of this class must be set.")

    # Replace with wrapped __init__.
    attrs_cls.__init__ = new_init_fn

    # Define convenience method for getting the set attribute.
    def which_oneof_attrib_name(self):
        attribs_with_value = [
            attrib for attrib in attribs if getattr(self, attrib.name) is not None
        ]

        if len(attribs_with_value) > 1:
            # Raise error if more than one attribute is set.
            raise ValueError("Only one attribute of this class can be set (not None).")

        if len(attribs_with_value) == 0:
            if must_be_set:
                # Raise error if none are set.
                raise ValueError("At least one attribute of this class must be set.")
            else:
                return None

        return attribs_with_value[0].name

    def which_oneof(self):
        attrib_name = self.which_oneof_attrib_name()

        if attrib_name is None:
            return None

        return getattr(self, attrib_name)

    attrs_cls.which_oneof_attrib_name = which_oneof_attrib_name
    attrs_cls.which_oneof = which_oneof

    return attrs_cls
