{ pkgs, lib, config, inputs, ... }:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  packages = let
    u = unstablePkgs;
  in
    [
    pkgs.git
    pkgs.gcc
    pkgs.cmake
    pkgs.ninja
    u.gcc-arm-embedded-13
    u.picotool
    pkgs.minicom
    pkgs.nodejs-slim
  ];

  languages.c.enable = true;
  languages.python = {
    enable = true;
    package = unstablePkgs.python312;
    uv.enable = true;
    uv.package = unstablePkgs.uv;
  };

  # https://devenv.sh/scripts/
  scripts = {
    setup_cmake = {
      exec = ''
        cd enV5
        cmake --preset unit_test
        cmake --preset env5_rev2_debug
        cmake --preset env5_rev2_release
      '';
      package = pkgs.bash;
      description = "setup cmake";
    };
    build_pico_release = {
      exec = ''
        cd enV5
        if [ -z "$1" ]; then
          cmake --build --preset env5_rev2_release --target "$1"
        else
          cmake --build --preset env5_rev2_release
        fi
      '';
      package = pkgs.bash;
      description = "build pico target of type RELEASE";
    };
    build_pico_debug = {
      exec = ''
        cd enV5
        if [ -z "$1" ]; then
          cmake --build --preset env5_rev2_debug --target "$1"
        else
          cmake --build --preset env5_rev2_debug
        fi
      '';
      package = pkgs.bash;
      description = "build pico target of type DEBUG";
    };
    flash_node = {
      exec = ''
        if [ -e "$1" ]; then
          if [ -r "$1" ]; then
            if [[ "$1" == *.uf2 ]]; then
              picotool load -f "$1"
            else
              echo "Not a valid file type (UF2)!"
              exit 1
            fi
          else
            echo "Can't read file!"
            exit 1
          fi
        else
          echo "You must provide an existing file to load!"
          exit 1
        fi
      '';
      package = pkgs.bash;
      description = "flash given UF2 file to pico";
    };
    format_files = {
      exec = ''
        INCLUDE_REGEX="^.*\.((((c|C)(c|pp|xx|\+\+)?$)|((h|H)h?(pp|xx|\+\+)?$))|(ino|pde|proto|cu))$"
        SRC_DIRECTORIES=("./enV5/src" "./enV5/test")
        SRC_FILES=$(find $SRC_DIRECTORIES -name .git -prune -o -regextype posix-egrep -regex "$INCLUDE_REGEX" -print)
        for file in $SRC_FILES; do
          clang-format -i -Werror --style=file --fallback-style="llvm" $file
        done
      '';
      package = pkgs.bash;
      description = "apply clang-format to src,test directory";
    };
  };

  enterShell = ''
    Welcome Back!
  '';
}
