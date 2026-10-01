# mgo

This repository contains the original MoE-Infinity experiments and a clean
research runtime rebuild.

## Use this for the current multi-GPU offloading paper

**`mgo_v2/`**

The two legacy MoE-Infinity directories are preserved as historical code and
as a source of the low-level fixed-slot / H2D executor. They are not the
current control-plane implementation.

See:

- `mgo_v2/README.md`
- `mgo_v2/MIGRATION.md`
- `mgo_v2/IMPLEMENTATION_STATUS.md`

Development branch: `codex/mgo-v2-runtime-20261001`.
