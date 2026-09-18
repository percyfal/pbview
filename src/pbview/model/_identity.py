import hashlib

import xarray as xr


def _datatree_id(dt: xr.DataTree, schema_version: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    for node in dt.subtree:
        h.update(node.path.encode())
        for name, var in sorted(node.ds.variables.items()):
            h.update(name.encode())
            h.update(str(var.dims).encode())
            h.update(str(var.shape).encode())
            h.update(str(var.dtype).encode())
        # include coord values (small, cheap, uniquely identifying)
        for name, coord in sorted(node.ds.coords.items()):
            if coord.size < 10_000:
                h.update(coord.values.tobytes())
    h.update(f"v{schema_version}".encode())
    return h.hexdigest()


def datatree_id(dt: xr.DataTree, *, schema_version: int) -> str:
    return _datatree_id(dt, schema_version)
