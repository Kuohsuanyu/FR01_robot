{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShellNoCC {
  name = "pipzone";
  packages = with pkgs; [
    (python311.withPackages (ps: [
      ps.pip
      ps.virtualenv
    ]))
    libz
    stdenv.cc.cc.lib
    gnumake
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.libz
    pkgs.stdenv.cc.cc
  ];
  shellHook = ''
    alias vi='nvim'
    alias vim='nvim'

    source ./.venv/bin/activate || python -m venv .venv && source ./.venv/bin/activate
  '';
}
