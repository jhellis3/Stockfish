## Overview

  [![Build][build-badge]][build-link]
  [![License][license-badge]][license-link]
  <br>
  [![Release][release-badge]][release-link]
  [![Commits][commits-badge]][commits-link]
  <br>

Crystal is a UCI chess engine derived from [Stockfish](https://stockfishchess.org).

The official Stockfish github repository may be found [here](https://github.com/official-stockfish/Stockfish).

Crystal seeks to address the following issues where chess engines often have trouble:

    1) Positional or tactical blindness due to over reductions or over pruning.
    2) Draw blindess due to the move horizon (50 move rule).
    3) Displayed PV reliability.

A few examples of what Crystal can do:

    1) (Ba4+) 3B4/1r2p3/r2p1p2/bkp1P1p1/1p1P1PPp/p1P4P/PP1K4/3B4 w - -
    2) (Draw) 4K1bn/5p2/5PpB/6P1/3k4/8/8/7q w - -
    3) (f6)   rk6/pP1p2p1/B7/3K1P2/8/8/7b/8 w - -
    4) (Rd3)  8/5K2/3p3p/3p3P/pp1P4/rkb1R3/p1p3P1/n1B2B2 w - -
    5) (Nd7)  1N1K1b1r/P3pPp1/4k1P1/rp1pB1RN/q4RP1/8/p2pB1p1/1b6 w - -
    6) (b4)   3K4/1p1B4/bB1k4/rpR1p3/2ppppp1/8/RPPPPP2/r1n5 w - -
    7) (Nc8)  8/1p1pNpbk/1q1P4/pP2p2K/P3N3/4P1P1/3P4/8 w - -

## Files

This distribution of Crystal consists of the following files:

  * [README.md][readme-link], the file you are currently reading.

  * [Copying.txt][license-link], a text file containing the GNU General Public
    License version 3.

  * [AUTHORS][authors-link], a text file with the list of authors for the project.

  * [src][src-link], a subdirectory containing the full source code, including a
    Makefile that can be used to compile Crystal on Unix-like systems.

  * a file with the .nnue extension, storing the neural network for the NNUE
    evaluation. Binary distributions will have this file embedded.

## Terms of use

Stockfish is free and distributed under the
[**GNU General Public License version 3**][license-link] (GPL v3). Essentially,
this means you are free to do almost exactly what you want with the program,
including distributing it among your friends, making it available for download
from your website, selling it (either by itself or as part of some bigger
software package), or using it as the starting point for a software project of
your own.

The only real limitation is that whenever you distribute Stockfish in some way,
you MUST always include the license and the full source code (or a pointer to
where the source code can be found) to generate the exact binary you are
distributing. If you make any changes to the source code, these changes must
also be made available under GPL v3.


[authors-link]:       https://github.com/official-stockfish/Stockfish/blob/master/AUTHORS
[build-link]:         https://github.com/jhellis3/Stockfish/actions/workflows/stockfish.yml
[commits-link]:       https://github.com/jhellis3/Stockfish/commits/crystal
[issue-link]:         https://github.com/jhellis3/Stockfish/issues/new?assignees=&labels=&template=BUG-REPORT.yml
[discussions-link]:   https://github.com/jhellis3/Stockfish/discussions/new
[license-link]:       https://github.com/jhellis3/Stockfish/blob/master/Copying.txt
[readme-link]:        https://github.com/jhellis3/Stockfish/blob/master/README.md
[release-link]:       https://github.com/jhellis3/Stockfish/releases/latest
[src-link]:           https://github.com/jhellis3/Stockfish/tree/crystal/src
[stockfish128-logo]:  https://stockfishchess.org/images/logo/icon_128x128.png
[uci-link]:           https://backscattering.de/chess/uci/
[website-link]:       https://stockfishchess.org
[website-blog-link]:  https://stockfishchess.org/blog/

[build-badge]:        https://img.shields.io/github/actions/workflow/status/jhellis3/Stockfish/stockfish.yml?branch=crystal&style=for-the-badge&label=crystal&logo=github
[commits-badge]:      https://img.shields.io/github/commits-since/official-stockfish/Stockfish/latest?style=for-the-badge
[license-badge]:      https://img.shields.io/github/license/official-stockfish/Stockfish?style=for-the-badge&label=license&color=success
[release-badge]:      https://img.shields.io/github/v/release/jhellis3/Stockfish?style=for-the-badge&label=official%20release
[website-badge]:      https://img.shields.io/website?style=for-the-badge&down_color=red&down_message=Offline&label=website&up_color=success&up_message=Online&url=https%3A%2F%2Fstockfishchess.org
