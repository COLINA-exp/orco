# recolina
Reconstruction toolkit for the COLINA experiment

## Install with Nix

From the repository root, install the `orco` command into your user profile:

```sh
nix profile install github:COLINA-exp/recolina#
```

The installed command accepts a transformer name, for example:

```sh
orco --help
orco pmaps --help
```

The Nix package includes the SQLite database under `orco/database` and opens it
read-only. The flake currently provides the package for `x86_64-linux`.

## Tests

Run the test suite in the development environment:

```sh
nix develop --command pytest -q -W error
```

`nix flake check` also runs the test suite in an isolated Nix build.
