import sleap
import sys
import os
import inspect


def find_member(obj, member_name):
    def _find_member(obj, member_name):
        for name, member in inspect.getmembers(obj):
            if name == member_name:
                return member

    for member_name in member_name.split("."):
        obj_ = _find_member(obj, member_name)
        if obj_ is None:
            return obj_
        obj = obj_

    return obj


def find_source_file(obj, root_obj):
    # Get relative filename
    fn = os.path.relpath(
        inspect.getsourcefile(obj),
        start=os.path.dirname(os.path.dirname(root_obj.__file__))
    ).replace("\\", "/")
    return fn


def find_source_lines(obj):
    # Find line numbers
    source_code, from_line = inspect.getsourcelines(obj)
    to_line = from_line + len(source_code) - 1
    
    return from_line, to_line


def resolve(module, fullname):
    if fullname == "":
        # Submodule specified, just infer path from the module name.
        return module.replace(".", "/") + ".py"
    
    # Search for member within module.
    member = find_member(sys.modules[module], fullname)
    
    if member is None:
        # Member not found, so we won't be linking this.
        return None
    
    try:
        fn = find_source_file(member, sleap)
    except TypeError:
        # Could not get the source file from member, so we won't be linking this.
        return None
    from_line, to_line = find_source_lines(member)

    return f"{fn}#L{from_line}-L{to_line}"

