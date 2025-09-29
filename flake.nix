# =============================================================================
# This flake provides a Python development environment tooling.
# Legacy nix-shell support is available through the wrapper in `shell.nix`.
# =============================================================================
{
  description = "RECOLINA development environment";

  inputs = {
    # Version pinning is managed in flake.lock.
    # Upgrading can be done with `nix flake lock --update input <input-name>`
    #
    #    nix flake lock --update-input nixpkgs
    nixpkgs.url      = "github:nixos/nixpkgs/nixos-25.05"; # nix flake lock --update input nixpkgs
    flake-utils.url  = "github:numtide/flake-utils";
    # Support for legacy nix-shell
    flake-compat = {
      url   = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [
      "x86_64-linux"
    ]
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
            ];
          };

          python = pkgs.python313.withPackages (ps: with ps; [
            coverage
            hypothesis
            jupyter
            matplotlib
            numpy
            pandas
            pint
            polars
            pyflakes
            pymysql
            pytest
            pytest-instafail
            pytest-xdist
#            pyvista
            scikit-image
            scikit-learn
            scipy
            tables
            uncertainties
          ]);

          pyright = pkgs.pyright;

        in
          {
            devShell = pkgs.mkShell {
              name = "RECOLINA";

              packages = [
                python
                pyright
              ];

              # Shell configuration
              shellHook = ''
                export PS1="RECOLINA> "
              '';
            };
          }
      );
}
