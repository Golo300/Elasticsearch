{
  description = "Elastic-Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

# docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        python-with-packages = pkgs.python311.withPackages (ps: with ps; [
            elasticsearch
        ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "dev-shell";
          buildInputs = [
            python-with-packages
          ];

        };
      }
    );
}

