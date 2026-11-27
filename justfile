# Run these recipes inside `nix develop` (or the direnv environment).
set positional-arguments

# List available recipes.
default:
    @just --list

# Run tests, treating warnings as errors; accepts pytest arguments.
test *args:
    python -m pytest -q -W error "$@"

# Run the isolated Nix checks; accepts nix flake check arguments.
check *args:
    nix flake check "$@"

# Build the installable ORCO package; accepts nix build arguments.
build *args:
    nix build .#orco "$@"

# Run the checkout's CLI, not an installed copy.
run *args:
    python orco/bin/orco "$@"

# Generate PMAPs; pass the usual transformer arguments.
pmaps *args:
    python orco/bin/orco pmaps "$@"

# Generate DSTs; pass the usual transformer arguments.
dsts *args:
    python orco/bin/orco dsts "$@"
