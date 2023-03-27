## Overview

[![Build Status](https://github.com/official-stockfish/Stockfish/actions/workflows/stockfish.yml/badge.svg)](https://github.com/official-stockfish/Stockfish/actions)

Crystal is a UCI chess engine derived from [Stockfish](https://stockfishchess.org).

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
    7) (Draw - disable NNUE) k7/P7/8/P7/2KB4/2B1B3/1B1B1B2/B1B1B1B1 b - -
    8) (Nc8)  8/1p1pNpbk/1q1P4/pP2p2K/P3N3/4P1P1/3P4/8 w - -

[Stockfish][website-link] is a **free and strong UCI chess engine** derived from
Glaurung 2.1 that analyzes chess positions and computes the optimal moves.

Stockfish **does not include a graphical user interface** (GUI) that is required
to display a chessboard and to make it easy to input moves. These GUIs are
developed independently from Stockfish and are available online. **Read the
documentation for your GUI** of choice for information about how to use
Stockfish with it.

See also the Stockfish [documentation][wiki-usage-link] for further usage help.

## Files

This distribution of Stockfish consists of the following files:

  * [README.md][readme-link], the file you are currently reading.

  * [Copying.txt][license-link], a text file containing the GNU General Public
    License version 3.

  * [AUTHORS][authors-link], a text file with the list of authors for the project.

  * [src][src-link], a subdirectory containing the full source code, including a
    Makefile that can be used to compile Stockfish on Unix-like systems.

  * a file with the .nnue extension, storing the neural network for the NNUE
    evaluation. Binary distributions will have this file embedded.

## The UCI protocol

The [Universal Chess Interface][uci-link] (UCI) is a standard text-based protocol
used to communicate with a chess engine and is the recommended way to do so for
typical graphical user interfaces (GUI) or chess tools. Stockfish implements the
majority of its options.

Developers can see the default values for the UCI options available in Stockfish
by typing `./stockfish uci` in a terminal, but most users should typically use a
chess GUI to interact with Stockfish.

For more information on UCI or debug commands, see our [documentation][wiki-commands-link].

## Compiling Stockfish

Stockfish has support for 32 or 64-bit CPUs, certain hardware instructions,
big-endian machines such as Power PC, and other platforms.

On Unix-like systems, it should be easy to compile Stockfish directly from the
source code with the included Makefile in the folder `src`. In general, it is
recommended to run `make help` to see a list of make targets with corresponding
descriptions.

```
cd src
make -j build ARCH=x86-64-modern
```

Detailed compilation instructions for all platforms can be found in our
[documentation][wiki-compile-link].

## Contributing

### Donating hardware

Improving Stockfish requires a massive amount of testing. You can donate your
hardware resources by installing the [Fishtest Worker][worker-link] and viewing
the current tests on [Fishtest][fishtest-link].

### Improving the code

In the [chessprogramming wiki][programming-link], many techniques used in
Stockfish are explained with a lot of background information.
The [section on Stockfish][programmingsf-link] describes many features
and techniques used by Stockfish. However, it is generic rather than
focused on Stockfish's precise implementation.

The engine testing is done on [Fishtest][fishtest-link].
If you want to help improve Stockfish, please read this [guideline][guideline-link]
first, where the basics of Stockfish development are explained.

Discussions about Stockfish take place these days mainly in the Stockfish
[Discord server][discord-link]. This is also the best place to ask questions
about the codebase and how to improve it.

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
[build-link]:         https://github.com/official-stockfish/Stockfish/actions/workflows/stockfish.yml
[commits-link]:       https://github.com/official-stockfish/Stockfish/commits/master
[discord-link]:       https://discord.gg/GWDRS3kU6R
[issue-link]:         https://github.com/official-stockfish/Stockfish/issues/new?assignees=&labels=&template=BUG-REPORT.yml
[discussions-link]:   https://github.com/official-stockfish/Stockfish/discussions/new
[fishtest-link]:      https://tests.stockfishchess.org/tests
[guideline-link]:     https://github.com/glinscott/fishtest/wiki/Creating-my-first-test
[license-link]:       https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt
[programming-link]:   https://www.chessprogramming.org/Main_Page
[programmingsf-link]: https://www.chessprogramming.org/Stockfish
[readme-link]:        https://github.com/official-stockfish/Stockfish/blob/master/README.md
[release-link]:       https://github.com/official-stockfish/Stockfish/releases/latest
[src-link]:           https://github.com/official-stockfish/Stockfish/tree/master/src
[stockfish128-logo]:  https://stockfishchess.org/images/logo/icon_128x128.png
[uci-link]:           https://backscattering.de/chess/uci/
[website-link]:       https://stockfishchess.org
[website-blog-link]:  https://stockfishchess.org/blog/
[wiki-link]:          https://github.com/official-stockfish/Stockfish/wiki
[wiki-usage-link]:    https://github.com/official-stockfish/Stockfish/wiki/Download-and-usage
[wiki-compile-link]:  https://github.com/official-stockfish/Stockfish/wiki/Compiling-from-source
[wiki-commands-link]: https://github.com/official-stockfish/Stockfish/wiki/Commands
[worker-link]:        https://github.com/glinscott/fishtest/wiki/Running-the-worker

[build-badge]:        https://img.shields.io/github/actions/workflow/status/official-stockfish/Stockfish/stockfish.yml?branch=master&style=for-the-badge&label=stockfish&logo=github
[commits-badge]:      https://img.shields.io/github/commits-since/official-stockfish/Stockfish/latest?style=for-the-badge
[discord-badge]:      https://img.shields.io/discord/435943710472011776?style=for-the-badge&label=discord&logo=Discord
[fishtest-badge]:     https://img.shields.io/website?style=for-the-badge&down_color=red&down_message=Offline&label=Fishtest&up_color=success&up_message=Online&url=https%3A%2F%2Ftests.stockfishchess.org%2Ftests%2Ffinished
[license-badge]:      https://img.shields.io/github/license/official-stockfish/Stockfish?style=for-the-badge&label=license&color=success
[release-badge]:      https://img.shields.io/github/v/release/official-stockfish/Stockfish?style=for-the-badge&label=official%20release
[website-badge]:      https://img.shields.io/website?style=for-the-badge&down_color=red&down_message=Offline&label=website&up_color=success&up_message=Online&url=https%3A%2F%2Fstockfishchess.org
