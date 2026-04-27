{
  description = "A Nix-flake-based Rust development environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "https://flakehub.com/f/nix-community/fenix/0.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    { self, ... }@inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [
                inputs.self.overlays.default
              ];
            };
          }
        );
    in
    {
      overlays.default = final: prev: {
        rustToolchain =
          with inputs.fenix.packages.${prev.stdenv.hostPlatform.system};
          combine (
            with stable;
            [
              clippy
              rustc
              cargo
              rustfmt
              rust-src
            ]
          );
      };
      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              rustToolchain
              openssl
              pkg-config
              onnxruntime

              # reqwest TLS deps
              openssl.dev

              # base64 + reqwest need these at link time
              zlib
              stdenv.cc.cc.lib

              # ollama for local VLM inference
              ollama

              cargo-deny
              cargo-edit
              cargo-watch
              rust-analyzer
            ];

            env = {
              RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";

              ORT_STRATEGY = "system";
              ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";

              C_INCLUDE_PATH = "${pkgs.onnxruntime}/include";
              CPLUS_INCLUDE_PATH = "${pkgs.onnxruntime}/include";

              LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.onnxruntime
                pkgs.stdenv.cc.cc.lib
                pkgs.openssl
              ];

              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.onnxruntime
                pkgs.stdenv.cc.cc.lib
                pkgs.openssl
                pkgs.zlib
              ];

              NIX_LDFLAGS = "-L${pkgs.onnxruntime}/lib -L${pkgs.stdenv.cc.cc.lib}/lib";
              ORT_PREFER_DYNAMIC_LINK = "1";

              # tells reqwest where openssl lives
              OPENSSL_DIR = "${pkgs.openssl.dev}";
              OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
              OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
            };
          };
        }
      );
    };
}
