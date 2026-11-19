{
  description = "Development environment for cheri-benchplot";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          pythonEnv = pkgs.python314.withPackages (ps: with ps; [
            # Runtime dependencies
            beautysh
            cxxfilt
            gitpython
            hatchling
            jinja2
            marshmallow-dataclass
            matplotlib
            networkx
            numpy
            openpyxl
            pandas
            polars
            pyelftools
            pyqt6
            pyserial
            scipy
            seaborn
            sortedcontainers
            sqlalchemy
            sqlalchemy-utils
            tabulate
            termcolor
            typeguard
            typing-extensions
            typing-inspect
            xlsxwriter

            # Development & test dependencies
            pip
            pytest
            pytest-mock
            pytest-timeout
            pytest-xdist
            sphinx
            sphinx-rtd-theme
            virtualenv
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.boost
              pkgs.cmake
              pkgs.fontconfig
              pkgs.git
              pkgs.libarchive
              pkgs.liberation_ttf
              pkgs.ninja
              pkgs.openssh
              pkgs.pre-commit
              pkgs.texliveSmall
              pkgs.xz
              pkgs.zstd
            ];

            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              export FONTCONFIG_FILE="${pkgs.makeFontsConf { fontDirectories = [ pkgs.liberation_ttf ]; }}"
            '';
          };
        });
    };
}
