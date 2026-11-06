# =============================================================================
# This flake provides the ORCO command and its Python development environment.
# The package includes SQLite data and passive example configs under share/orco.
# =============================================================================
{
  description = "ORCO reconstruction toolkit and development environment";

  inputs = {
    # Version pinning is managed in flake.lock.
    # Update the lock deliberately and run the test/build checks afterwards.
    nixpkgs.url      = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url  = "github:numtide/flake-utils";
    # Compatibility input; no legacy shell.nix wrapper is currently provided.
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

          runtimePackages = ps: with ps; [
            numpy
            pandas
            pint
            tables
          ];

          runtimePython = pkgs.python313.withPackages runtimePackages;
          testPython = pkgs.python313.withPackages (ps: runtimePackages ps ++ [ ps.pytest ]);

          pythonSitePackages = pkgs.python313.sitePackages;

          orco = pkgs.stdenvNoCC.mkDerivation {
            pname = "orco";
            version = "0.1.0";
            src = self;
            dontUnpack = true;
            dontBuild = true;
            nativeBuildInputs = [ pkgs.makeWrapper ];

            installPhase = ''
              runHook preInstall

              mkdir -p "$out/bin" "$out/${pythonSitePackages}/orco"
              cp -R "$src/orco/." "$out/${pythonSitePackages}/orco/"

              makeWrapper "${runtimePython}/bin/python" "$out/bin/orco" \
                --prefix PYTHONPATH : "$out/${pythonSitePackages}" \
                --add-flags "$out/${pythonSitePackages}/orco/bin/orco"

              "$out/bin/orco" --help >/dev/null
              "$out/bin/orco" pmaps --help >/dev/null
              PYTHONPATH="$out/${pythonSitePackages}" "${runtimePython}/bin/python" -c \
                'from orco.database.io import get_database; from orco.internal.symbols import Detector; assert get_database(Detector.pColina).is_file()'

              runHook postInstall
            '';

            meta = {
              description = "Reconstruction toolkit for the COLINA experiment";
              mainProgram = "orco";
              platforms = [ "x86_64-linux" ];
            };
          };

          pyright = pkgs.pyright;

        in
          {
            checks.pytest = pkgs.runCommand "orco-tests" {
              nativeBuildInputs = [ testPython ];
            } ''
              cp -R ${self}/orco ./orco
              chmod -R u+w ./orco
              pytest -q -W error orco
              touch "$out"
            '';

            packages = {
              default = orco;
              inherit orco;
            };

            apps = {
              default = {
                type = "app";
                program = "${orco}/bin/orco";
              };
              orco = {
                type = "app";
                program = "${orco}/bin/orco";
              };
            };

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
