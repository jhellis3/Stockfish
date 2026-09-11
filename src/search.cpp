/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <list>
#include <ratio>
#include <string>
#include <utility>
#include "bitboard.h"
#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "syzygy/tbprobe.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

static constexpr std::array<int, 16> lmrDivisor = {3637, 2787, 2761, 2939, 3171, 3347, 3147, 2762,
                                                   2772, 3106, 3107, 3060, 3112, 2991, 3090, 3542};

namespace TB = Tablebases;

using namespace Search;

namespace {

constexpr u64 NODES_LIMIT_OUTPUT = 10'000'000;

constexpr int SEARCHEDLIST_CAPACITY = 32;
using SearchedList                  = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

// (*Scaler) All tuned parameters at time controls shorter than
// optimized for require verifications at longer time controls.

int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us     = pos.side_to_move();
    const auto  m      = (ss - 1)->currentMove;
    const auto& shared = w.sharedHistory;
    const int   pcv    = shared.pawn_correction_entry(pos)[us].pawn;
    const int   micv   = shared.minor_piece_correction_entry(pos)[us].minor;
    const int   wnpcv  = shared.nonpawn_correction_entry<WHITE>(pos)[us].nonPawnWhite;
    const int   bnpcv  = shared.nonpawn_correction_entry<BLACK>(pos)[us].nonPawnBlack;
    const int   cntcv =
      m.is_ok()
          ? 8761
            * ((*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
               + (*(ss - 4)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()])
          : 64049;

    return 15341 * pcv + 10569 * micv + 12906 * (wnpcv + bnpcv) + cntcv;
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
Value to_corrected_static_eval(const Value v, const int cv, const uint8_t r50) {
    return std::clamp((v + cv / 131072) * (100 - r50) / 100, -VALUE_MAX_EVAL + 1, VALUE_MAX_EVAL - 1);
}

void update_correction_history(const Position& pos,
                               Stack* const    ss,
                               Search::Worker& workerThread,
                               const int       bonus) {
    const Move  m  = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    constexpr int nonPawnWeight = 186;
    auto&         shared        = workerThread.sharedHistory;

    shared.pawn_correction_entry(pos)[us].pawn << bonus;
    shared.minor_piece_correction_entry(pos)[us].minor << bonus * 150 / 128;
    shared.nonpawn_correction_entry<WHITE>(pos)[us].nonPawnWhite << bonus * nonPawnWeight / 128;
    shared.nonpawn_correction_entry<BLACK>(pos)[us].nonPawnBlack << bonus * nonPawnWeight / 128;

    if (m.is_ok())
    {
        const Square to = m.to_sq();
        const Piece  pc = pos.piece_on(to);
        (*(ss - 2)->continuationCorrectionHistory)[pc][to] << bonus * 130 / 128;
        (*(ss - 4)->continuationCorrectionHistory)[pc][to] << bonus * 70 / 128;
    }
}

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(
   const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove,
                      bool            PvNode);

}  // namespace

Search::Worker::Worker(SharedState&                   sharedState,
                       std::unique_ptr<SearchManager> sm,
                       usize                          threadId,
                       usize                          numaThreadId,
                       usize                          numaTotalThreads,
                       NumaReplicatedAccessToken      token) :
    // Unpack the SharedState struct into member variables
    sharedHistory(sharedState.sharedHistories.at(token.get_numa_index())),
    continuationHistory(sharedHistory.continuationHistory()),
    threadIdx(threadId),
    numaThreadIdx(numaThreadId),
    numaTotal(numaTotalThreads),
    numaAccessToken(token),
    manager(std::move(sm)),
    options(sharedState.options),
    threads(sharedState.threads),
    tt(sharedState.tt),
    network(sharedState.network),
    refreshTable(network[token]) {
    clear();
}

void Search::Worker::ensure_network_replicated() {
    // Access once to force lazy initialization, avoiding initialization during search
    (void) (network[numaAccessToken]);
}

void Search::Worker::start_searching() {

    accumulatorStack.reset();

    // Non-main threads go directly to iterative_deepening()
    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
    tt.new_search();
    main_manager()->updates.onStart();

    if (rootMoves.empty())
    {
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
        main_manager()->updates.onBestmove(UCIEngine::move(Move::none()), "");
        return;
    }

    // Main thread starts non-main threads, and begins own search
    threads.start_searching();
    bool uciPvSent = iterative_deepening();

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.
    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {}

    // Stop the threads if not already stopped (also raise the stop if "ponderhit"
    // just reset threads.ponder).
    threads.stop = true;

    // Wait until all threads have finished
    threads.wait_for_search_finished();

    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);

    Worker* bestThread = this;

    if (!limits.depth)
        bestThread = threads.get_best_thread()->worker.get();

    main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;

    if (bestThread->rootMoves[0].pv.size() == 1
        && bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
        uciPvSent = false;

    // Send PV info if it has changed since last output in iterative_deepening()
    if (!uciPvSent || bestThread != this)
        main_manager()->output_pv(*bestThread, threads, tt, bestThread->rootDepth);

    // In rare cases, output_pv() may change the ponder move through syzygy_extend_pv()
    std::string ponder;
    if (bestThread->rootMoves[0].pv.size() > 1)
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());
    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Main iterative deepening loop. It calls search() repeatedly with increasing
// depth until the allocated thinking time has been consumed, the user stops
// the search, or the maximum search depth is reached.
bool Search::Worker::iterative_deepening() {

    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

    PVMoves pv;

    RootPVMoves lastBestMovePV;
    Value       lastBestMoveScore = -VALUE_INFINITE;

    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
    double totBestMoveChanges = 0;
    int    delta, iterIdx                        = 0;

    // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
    // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
    // (ss + 2) is needed for initialization of cutOffCnt.
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &continuationHistory[0][0][NO_PIECE][0];  // Use as a sentinel
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = &pv;

    if (mainThread)
    {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    size_t multiPV = size_t(options["MultiPV"]);

    int rootContempt = UCIEngine::to_int(int(options["Contempt"]), rootPos);

    multiPV = std::min(multiPV, rootMoves.size());

    bool uciPvSent = false;

    lowPlyHistory.fill(102);

    for (Color c : {WHITE, BLACK})
        for (int i = 0; i < UINT_16_HISTORY_SIZE; i++)
            mainHistory[c][i] = mainHistory[c][i] * 729 / 1024;

    // Iterative deepening loop until requested to stop or the target depth is reached
    while (rootDepth + 1 < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth >= limits.depth))
    {
        rootDepth++;

        // Age out PV variability metric and signal the start of a new iteration
        if (mainThread)
        {
            totBestMoveChanges /= 2;
            uciPvSent = false;
        }

        // Save the last iteration's scores before the first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.
        for (usize i = 0; i < rootMoves.size(); ++i)
        {
            rootMoves[i].previousScore      = rootMoves[i].score;
            rootMoves[i].previousPV         = rootMoves[i].pv;
            rootMoves[i].previousScoreExact = i < multiPV;
        }

        usize pvFirst = pvLast = 0;

        // MultiPV loop: we perform a full root search for each PV line
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }

            lastIterationIdxPV = rootMoves[pvIdx].previousPV;

            // Reset UCI info selDepth for each depth and each PV line
            selDepth = 0;

            // Reset aspiration window starting size
            Value avg   = rootMoves[pvIdx].averageScore;
            int momentum = (int(avg) * avg) >> 13;
            delta        = 9;

            // Dynamic symmetric contempt. If we at least have a draw, we have contempt; otherwise assume opponent has it.
            if (rootMoves[0].averageScore >= VALUE_DRAW)
            {
                contempt[us] = rootContempt;
                contempt[~us] = 0;
            }
            else
            {
                contempt[us] = 0;
                contempt[~us] = rootContempt;
            }

            alpha = std::max(avg - (delta + (avg < 0 ? momentum : 0)),-VALUE_INFINITE);
            beta  = std::min(avg + (delta + (avg > 0 ? momentum : 0)), VALUE_INFINITE);

            // Start with a small aspiration window and, in the case of a fail
            // high/low, enlarge the window progressively.
            while (true)
            {
                rootDelta = beta - alpha;
                bestValue = search<Root>(rootPos, ss, alpha, beta, rootDepth, false);

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one is set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in the case of MultiPV
                // search the already searched PV lines are preserved.
                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                // If search has been stopped, we break immediately. Sorting is
                // safe because RootMoves is still valid, although it refers to
                // the previous iteration.
                if (threads.stop)
                    break;

                // When failing high/low give some update before a re-search. To avoid
                // excessive output that could hang GUIs like Fritz 19, only start
                // at nodes > 10M (rather than depth N, which can be reached quickly).
                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > NODES_LIMIT_OUTPUT)
                    main_manager()->output_pv(*this, threads, tt, rootDepth);

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = alpha;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                  if (mainThread)
                      mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {
                    alpha = std::max(beta - delta, alpha); // ???
                    beta  = std::min(bestValue + delta, VALUE_INFINITE);
                }
                else
                    break;

                delta += 47 * delta / 128;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            if (threads.stop && pvIdx)
            {
                // In multiPV analysis we do not let aborted searches spoil
                // mated-in/TB loss scores from a completed search in an earlier
                // PV line. Hence we guard against an aborted pvIdx line overtaking
                // pvIdx - 1 when pvIdx - 1 is a proven loss. Moreover, we do not
                // trust an exact loss score from an aborted search.
                if ((is_loss(rootMoves[pvIdx - 1].score) && rootMoves[pvIdx] < rootMoves[pvIdx - 1])
                    || rootMoves[pvIdx].is_exact_loss())
                {
                    // If previousScore is exact and worse than pvIdx - 1, we
                    // can safely use it. If it is equal, we make sure it cannot
                    // overtake pvIdx - 1.
                    if (rootMoves[pvIdx].previousScore != -VALUE_INFINITE
                        && rootMoves[pvIdx].previousScoreExact
                        && rootMoves[pvIdx].previousScore <= rootMoves[pvIdx - 1].score)
                    {
                        rootMoves[pvIdx].score = rootMoves[pvIdx].uciScore =
                          rootMoves[pvIdx].previousScore;
                        rootMoves[pvIdx].previousScore = -VALUE_INFINITE;
                        rootMoves[pvIdx].pv            = rootMoves[pvIdx].previousPV;
                        rootMoves[pvIdx].unset_inexact();
                    }

                    // Otherwise, if we can, we cap the score to the best possible, and mark
                    // the score as inexact (also a valid excuse for the incomplete PV).
                    else
                    {
                        if (is_loss(rootMoves[pvIdx - 1].score))
                        {
                            rootMoves[pvIdx].score = rootMoves[pvIdx].uciScore =
                              rootMoves[pvIdx - 1].score;
                            rootMoves[pvIdx].previousScore = -VALUE_INFINITE;
                            rootMoves[pvIdx].pv.resize(1);
                            rootMoves[pvIdx].inexactUpper = true;
                        }
                        else
                            rootMoves[pvIdx].inexactUpper = false;

                        rootMoves[pvIdx].inexactLower = !rootMoves[pvIdx].inexactUpper;
                    }
                }

                // Finally, we mark all loss scores from partially searched moves as inexact.
                for (usize i = pvIdx + 1; i < multiPV; ++i)
                    if (rootMoves[i].is_exact_loss())
                        rootMoves[i].inexactLower = true;
            }

            // Sort the PV lines searched so far and update the GUI
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

            if (mainThread && !threads.stop && (pvIdx + 1 == multiPV || nodes > NODES_LIMIT_OUTPUT))
            {
                main_manager()->output_pv(*this, threads, tt, rootDepth);
                uciPvSent = (pvIdx + 1 == multiPV);
            }

            if (threads.stop)
                break;
        }

        const bool forgottenMate = lastBestMoveScore != -VALUE_INFINITE
                                && is_mate_or_mated(lastBestMoveScore)
                                && (std::abs(rootMoves[0].score) < std::abs(lastBestMoveScore)
                                    || rootMoves[0].is_inexact());

        if (!threads.stop)
        {
            // Do not replace (shorter) mate scores from a previous iteration
            if (!forgottenMate)
            {
                lastBestMovePV    = rootMoves[0].pv;
                lastBestMoveScore = rootMoves[0].score;
            }
        }

        const bool abortedLossSearch = threads.stop && !pvIdx && rootMoves[0].is_exact_loss();

        // An exact mated-in/TB-loss score from an aborted search cannot be
        // trusted: the loss could be delayed or refuted upon exploring the
        // remaining root-moves. Thus here we roll back to the score from the
        // previous iteration. We do the same if a search has failed to recover
        // a mate score that was found in a previous iteration.
        if (abortedLossSearch || (rootMoves[0].score != -VALUE_INFINITE && forgottenMate))
        {
            // Bring the last best move to the front for best thread selection
            if (!lastBestMovePV.empty())
            {
                Utility::move_to_front(rootMoves, [&lastPV = std::as_const(lastBestMovePV)](
                                                    const auto& rm) { return rm == lastPV[0]; });
                rootMoves[0].score = rootMoves[0].uciScore = lastBestMoveScore;
                rootMoves[0].pv                            = lastBestMovePV;
                rootMoves[0].unset_inexact();

                if (mainThread)
                    uciPvSent = false;
            }
            // For an aborted d1 search we label the loss score as inexact
            else if (abortedLossSearch)
                rootMoves[0].inexactLower = true;
        }

        // Have we found a "mate in x" after a completed iteration?
        if (limits.mate && !threads.stop && is_mate_or_mated(rootMoves[0].score)
            && VALUE_MATE - std::abs(rootMoves[0].score) <= 2 * limits.mate)
            threads.stop = true;

        if (!mainThread)
            continue;

        // Use part of the gained time from a previous stable move for the current move
        for (auto&& th : threads)
        {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
        {
            double fallingEval = (11.48 + 2.30 * (mainThread->bestPreviousAverageScore - bestValue)
                                     +  1.1 * (mainThread->iterValue[iterIdx] - bestValue)) / 100.0;

            fallingEval = std::clamp(fallingEval, 0.67, 1.70);

            double bestMoveInstability = 0.67 + 2 * totBestMoveChanges / threads.size();
            auto elapsedT = elapsed();
            auto optimumT = mainThread->tm.optimum();
            double maximumT = mainThread->tm.maximum();

            // Stop the search if we have only one legal move, or if available time elapsed
            if (   (rootMoves.size() == 1 && (elapsedT > optimumT / 16))
                || elapsedT > std::min(4.33 * optimumT, maximumT) // review
                || elapsedT > optimumT * fallingEval * bestMoveInstability)
            {
                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return false;

    return uciPvSent;
}


void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st, Stack* const ss) {
    do_move(pos, move, st, pos.gives_check(move), ss);
}

void Search::Worker::do_move(
  Position& pos, const Move move, StateInfo& st, const bool givesCheck, Stack* const ss) {

    // prefetch_key() does not model castling, en passant or promotion exactly.
    // The correction-history prefetches also approximate castling and promotion.
    // For these rare moves the prefetches land on unused lines.
    prefetch(tt.first_entry(pos.prefetch_key(move)));

    bool capture = pos.capture_stage(move);

    if (ss != nullptr)
    {
        const Piece  pc = pos.moved_piece(move);
        const Square to = move.to_sq();

        prefetch(&(*(ss - 1)->continuationCorrectionHistory)[pc][to]);
        prefetch(&(*(ss - 3)->continuationCorrectionHistory)[pc][to]);
    }

    ++nodes;

    Dirties& dirties = accumulatorStack.push();
    pos.do_move(move, st, givesCheck, dirties, &tt, &sharedHistory);

    if (ss != nullptr)
    {
        auto& dirtyPiece = dirties.dirtyPiece;
        ss->currentMove  = move;
        ss->continuationHistory =
          &continuationHistory[ss->inCheck][capture][dirtyPiece.pc][move.to_sq()];
        ss->continuationCorrectionHistory =
          &continuationCorrectionHistory[dirtyPiece.pc][move.to_sq()];
    }
}

void Search::Worker::do_null_move(Position& pos, StateInfo& st, Stack* const ss) {
    pos.do_null_move(st);
    ss->currentMove                   = Move::null();
    ss->continuationHistory           = &continuationHistory[0][0][NO_PIECE][0];
    ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
}

void Search::Worker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);
    accumulatorStack.pop();
}

void Search::Worker::undo_null_move(Position& pos) { pos.undo_null_move(); }


// Reset histories, usually before a new game
void Search::Worker::clear() {
    mainHistory.fill(-5);
    captureHistory.fill(-742);

    // Each thread clears its part of the dynamically-sized shared histories.
    // The constant-size continuation history is initialized by thread 0 of each NUMA node.
    sharedHistory.correctionHistory.clear_range(-5, numaThreadIdx, numaTotal);
    sharedHistory.pawnHistory.clear_range(-1338, numaThreadIdx, numaTotal);

    if (numaThreadIdx == 0)
        for (bool inCheck : {false, true})
            for (StatsType c : {NoCaptures, Captures})
                for (auto& to : continuationHistory[inCheck][c])
                    for (auto& h : to)
                        h.fill(-586);

    ttMoveHistory = 0;

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(5);

    for (usize i = 1; i < reductions.size(); ++i)
        reductions[i] = int(2872 / 128.0 * std::log(i));

    refreshTable.clear(network[numaAccessToken]);
}


// Main search function for both PV and non-PV nodes
template<NodeType nodeType>
Value Search::Worker::search(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, const bool cutNode) {

    constexpr bool PvNode   = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;
    const bool     allNode  = !(PvNode || cutNode);

    // Dive into quiescence search when the depth reaches zero
    if (depth <= 0)
        return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta);

    // Limit the depth if extensions made it too large
    depth = std::min(depth, MAX_PLY - 1);

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    PVMoves   pv;
    StateInfo st;

    Key     posKey;
    Move    move, excludedMove, bestMove;
    Depth   extension, newDepth;
    Value   bestValue, value, eval, probCutBeta, drawValue;
    bool    givesCheck, improving, priorCapture, isMate, gameCycle;
    bool    capture, opponentWorsening,
            ttCapture, kingDanger, ourMove, nullParity;
    Piece   movedPiece;
    uint8_t rule50;

    SearchedList capturesSearched;
    SearchedList quietsSearched;

    // Step 1. Initialize node
    Worker* thisThread  = this;
    ss->inCheck         = pos.checkers();
    priorCapture        = pos.captured_piece();
    Color us            = pos.side_to_move();
    ss->moveCount       = 0;
    bestValue           = -VALUE_INFINITE;
    gameCycle           = kingDanger = false;
    ourMove             = !(ss->ply & 1);
    nullParity          = (ourMove == thisThread->nmpSide);
    ss->secondaryLine   = false;
    ss->mainLine        = false;
    drawValue           = contempt[us];
    rule50              = std::min(90, pos.rule50_count());;

    ss->followPV = rootNode
                || ((ss - 1)->followPV
                    && (static_cast<usize>(ss->ply - 1) < lastIterationIdxPV.size()
                        && (ss - 1)->currentMove == lastIterationIdxPV[ss->ply - 1]));

    // Check for the available remaining time
    if (is_mainthread())
        main_manager()->check_time(*this);

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    excludedMove = ss->excludedMove;

    if (!rootNode)
    {
        // Check if we have an upcoming move which draws by repetition, or
        // if the opponent had an alternative move earlier to this position.
        if (pos.upcoming_repetition(ss->ply) && !excludedMove)
        {
            if (drawValue >= beta)
                return drawValue;

            gameCycle = true;
            alpha = std::max(alpha, drawValue);
        }

        // Step 2. Check for aborted search or immediate draw
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : drawValue;

        // Step 3. Mate distance pruning. Even if we mate at the next move our score
        // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
        // a shorter mate was found upward in the tree then there is no need to search
        // because we will never beat the current alpha. Equal and opposite logic applies
        // when being mated. In either case, return a fail-high score.
        if (alpha >= mate_in(ss->ply+1))
            return mate_in(ss->ply+1);
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    (ss - 1)->reduction        = 0;
    ss->statScore              = 0;
    (ss + 2)->cutoffCnt        = 0;
    (ss + 1)->priorNMPFailHigh = 0;

    const auto correctionValue = correction_value(*this, pos, ss);

    // Step 4. Transposition table lookup
    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);

    ss->ttHit    = ttHit;
    ttData.move  = rootNode ? rootMoves[pvIdx].pv[0] : ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply) : VALUE_NONE;
    ttData.value = (abs(ttData.value) > VALUE_MAX_EVAL) ? ttData.value : ttData.value * (100 - rule50) / (100 - ttData.rule50);
    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture    = ttData.move && pos.capture_stage(ttData.move);

    // At non-PV nodes we check for an early TT cutoff
    if (  !PvNode
        && !excludedMove
        && !gameCycle
        && !(ss-1)->mainLine
        && ttData.depth > depth - (ttData.value < beta)
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 4))
    {
        // If the ttMove is quiet, update move sorting heuristics on TT hit
        if (ttData.move && ttData.value >= beta)
        {
            // Bonus for a quiet ttMove that fails high
            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move, 131 * depth);

            // Extra penalty for early quiet moves of the previous ply
            if (prevSq != SQ_NONE && (ss - 1)->moveCount < 5 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2210);
        }

        // Partial workaround for the graph history interaction problem.
        // For high rule50 counts don't produce transposition table cutoffs.
        if (depth >= 7 && ttData.move && pos.pseudo_legal(ttData.move) && pos.legal(ttData.move)
            && !is_decisive(ttData.value))
        {
            pos.do_move(ttData.move, st);
            Key nextPosKey                             = pos.key();
            auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
            pos.undo_move(ttData.move);

            if (!is_valid(ttDataNext.value))
                return ttData.value;

            if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                return ttData.value;
        }
        else
            return ttData.value;

    }  // No cutoff, but why? Compare the aspiration window to the inexact bound
    else if (!PvNode && !excludedMove && ttData.depth > depth - (ttData.value <= beta)
             && is_valid(ttData.value) && ttData.bound != BOUND_EXACT
             && ttData.bound & (ttData.value >= beta ? BOUND_UPPER : BOUND_LOWER) && depth > 5)
    {
        // If such a mismatch is the only reason cutoff failed, the TT entry is now useless
        ttWriter.penalize(1);
    }

    // Step 7. Tablebases probe
    if (!rootNode && !excludedMove && tbConfig.cardinality)
    {
        int piecesCount = popcount(pos.pieces());

        if (    piecesCount <= tbConfig.cardinality
            &&  rule50 == 0
            && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore wdl = TB::probe_wdl(pos, &err);

            // Force check of time on the next occasion
            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                ++tbHits;

                int drawScore = tbConfig.useRule50 ? 1 : 0;

                int centiPly = tbConversionFactor * ss->ply / 100;

                Value tbValue =    wdl < -drawScore ? -VALUE_TB_WIN + (10 * tbConversionFactor * (wdl == -1)) + centiPly + tbConversionFactor * popcount(pos.pieces( pos.side_to_move()))
                                 : wdl >  drawScore ?  VALUE_TB_WIN - (10 * tbConversionFactor * (wdl ==  1)) - centiPly - tbConversionFactor * popcount(pos.pieces(~pos.side_to_move()))
                                 : wdl < 0 ? Value(-56) : drawValue;

                if (    abs(wdl) <= drawScore
                    || !ss->ttHit
                    || (wdl < -drawScore &&  beta > tbValue + 9)
                    || (wdl >  drawScore && alpha < tbValue - 9))
                {
                    ttWriter.write(posKey, tbValue, ss->ttPv, wdl > drawScore ? BOUND_LOWER : wdl < -drawScore ? BOUND_UPPER : BOUND_EXACT,
                                   depth, Move::none(), VALUE_NONE, tt.generation(), rule50);

                    return tbValue;
                }
            }
        }
    }

    kingDanger = !ourMove && pos.king_danger(us);

    // Step 6. Static evaluation of the position
    Value      unadjustedStaticEval = VALUE_NONE;
    //const auto correctionValue      = correction_value(*this, pos, ss);
    if (ss->inCheck)
    {
        // Skip early pruning when in check
        ss->staticEval = eval = (ss - 2)->staticEval;
        improving             = false;
    }
    else
    {
    if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue, rule50);

        // ttValue can be used as a better position evaluation
        if (   is_valid(ttData.value)
            && ttData.move != Move::none()
            && ttData.value > eval
            && ttData.bound & BOUND_LOWER)
            eval = ttData.value;

        else if (   !ourMove
                 && ttData.value < eval
                 && ttData.bound & BOUND_UPPER)
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue, rule50);

        // Static evaluation is saved as it was before adjustment by correction history
        ttWriter.write(posKey, VALUE_NONE, false, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, tt.generation(), rule50);
    }

    // Use static evaluation difference to improve quiet move ordering
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture)
    {
        int evalDiff = std::clamp(-int((ss - 1)->staticEval + ss->staticEval), -189, 194) + 60;
        mainHistory[~us][((ss - 1)->currentMove).raw()] << evalDiff * 11;
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            sharedHistory.pawn_entry(pos)[pos.piece_on(prevSq)][prevSq] << evalDiff * 13;
    }

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.
    improving         = ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

    // Begin early pruning.
    if (   !PvNode
        && !thisThread->nmpGuardV
        && !is_decisive(eval)
        && !is_decisive(beta)
        &&  eval >= beta)
    {
        Value futilityMult = std::min(45 + depth * 4, 85);
        futilityMult -= 20 * !ss->ttHit;

        Value futilityMargin = futilityMult * depth
                             - (2789 * improving + 335 * opponentWorsening) * futilityMult / 1024
                             + std::abs(correctionValue) / 198435;

       // Step 8. Futility pruning: child node (~40 Elo)
       // The depth condition is important for mate finding.
       if (    depth < (9 - 2 * ((ss-1)->mainLine || (ss-1)->secondaryLine || (ttData.move && !ttCapture)))
           && !ss->ttPv
           && !kingDanger
           && !excludedMove
           && !gameCycle
           && !(thisThread->nmpGuard && nullParity)
           &&  eval - futilityMargin >= beta)
           return (661 * beta + 363 * eval) / 1024; // review

       // Step 9. Null move search with verification search (~35 Elo)
       if (   !thisThread->nmpGuard
           &&  cutNode
           && !gameCycle
           && !excludedMove
           &&  eval >= ss->staticEval
           &&  ss->staticEval + 50 * ss->priorNMPFailHigh >= beta - 13 * depth - 47 * improving + 365
           &&  pos.non_pawn_material(us)
           && !kingDanger
           && (rootDepth < 11 || ourMove || MoveList<LEGAL>(pos).size() > 5))
       {
           assert(eval - beta >= 0);

           thisThread->nmpSide = ourMove;

           // Null move dynamic reduction based on depth and eval
           Depth R = 7 + depth / 3 + std::max((ss->staticEval - beta) / 256, 0);

           if (!ourMove && (ss-1)->secondaryLine)
               R = std::min(R, 8);

           if (   depth < 11
               || ttData.value >= beta
               || ttData.depth < depth-R
               || !(ttData.bound & BOUND_UPPER))
           {
              ss->currentMove                   = Move::null();
              ss->continuationHistory           = &continuationHistory[0][0][NO_PIECE][0];
              ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];

              do_null_move(pos, st, ss);
              thisThread->nmpGuard = true;
              Value nullValue = -search<NonPV>(pos, ss+1, -beta, -beta+1, depth-R, false);
              thisThread->nmpGuard = false;
              undo_null_move(pos);

              if (nullValue >= beta)
              {
                  // Verification search
                  thisThread->nmpGuardV = true;
                  Value v = search<NonPV>(pos, ss, beta-1, beta, depth-R, false);
                  thisThread->nmpGuardV = false;

                  // While it is unsafe to return mate scores from null search, mate scores
                  // from verification search are fine.
                  if (v >= beta)
                  {
                      ++ss->priorNMPFailHigh;
                      return is_win(v) ? v : std::min(nullValue, VALUE_MAX_EVAL);
                  }
              }
           }
       }

       improving |= ss->staticEval >= beta;

       probCutBeta = beta + 241 - 64 * improving;
       // Step 12. ProbCut
       // If we have a good enough capture (or queen promotion) and a reduced search
       // returns a value much above beta, we can (almost) safely prune the previous move.
       if (    depth > 4
           && (ttCapture || !ttData.move)
           // If we don't have a ttHit or our ttDepth is not greater our
           // reduced depth search, continue with the probcut.
           && (!ss->ttHit || (ttData.depth < depth - 3 && ttData.value >= probCutBeta && ttData.value != VALUE_NONE)))
       {
           assert(probCutBeta < VALUE_INFINITE);
           MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
           Depth      probCutDepth = depth - (improving ? 5 : 3);

           while ((move = mp.next_move()) != Move::none())
               if (move != excludedMove)
               {
                   assert(pos.capture_stage(move));

                   movedPiece = pos.moved_piece(move);

                   do_move(pos, move, st, ss);

                   value = -search<NonPV>(pos, ss+1, -probCutBeta, -probCutBeta+1, probCutDepth, !cutNode);

                   undo_move(pos, move);

                   if (value >= probCutBeta)
                   {
                       if (!excludedMove)
                           ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv,
                                     BOUND_LOWER, probCutDepth + 1, move, unadjustedStaticEval, tt.generation(), rule50);

                       return value;
                   }
               }
       }
    } // End early Pruning

    // Step 11. Internal iterative reductions
    // For PV nodes without a ttMove as well as for deep enough cutNodes, we decrease depth.
    // (*Scaler) Especially if they make IIR less aggressive.
    if (   !ss->followPV
        && !allNode
        &&  depth >= 6
        && !ttData.move
        && !gameCycle
        && (!PvNode || !(ss-1)->mainLine || (ss-1)->moveCount > 1)
        && !(ss-1)->secondaryLine)
        depth -= 2;

    } // In check search starts here

   // Step 12. A small Probcut idea
   probCutBeta = beta + 428;
   if (     ss->inCheck
        && !PvNode
        &&  ttCapture
        &&  ourMove
        && !gameCycle
        && !excludedMove
        && !kingDanger
        && !(ss-1)->secondaryLine
        && !(thisThread->nmpGuard && nullParity)
        && !(thisThread->nmpGuardV && nullParity)
        && (ttData.bound & BOUND_LOWER)
        && ttData.depth >= depth - 4
        && ttData.value >= probCutBeta
        && !is_decisive(ttData.value)
        && !is_decisive(beta))
        return probCutBeta;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};


    MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, contHist,
                  &sharedHistory, ss->ply);

    value = bestValue;

    int moveCount = 0;

    bool allowExt = (depth + ss->ply + 2 < MAX_PLY) && (ss->ply < 2 * rootDepth);

    bool lmrCapture = cutNode && (ss-1)->moveCount > 1;

    bool gameCycleExtension =    gameCycle
                              && (   PvNode
                                  || ((ss-1)->secondaryLine && pvValue < drawValue));

    bool kingDangerThem = ourMove && pos.king_danger(~us);

    bool doSingular =    !rootNode
                      && !excludedMove // Avoid recursive singular search
                      &&  is_valid(ttData.value)
                      && (ttData.bound & BOUND_LOWER)
                      && !is_loss(alpha)
                      && !is_decisive(ttData.value)
                      &&  ttData.depth >= depth - 3
                      &&  depth >= 6 + ss->ttPv;

    int lmrAdjustment =   cutNode * 3 // 3 + !ttData.move
                        + ttCapture
                        + ((ss+1)->cutoffCnt > 3)
                        - (!PvNode && doSingular)
                        - (ss->ttPv * (1 + (ttData.value > alpha) + (ttData.depth >= depth)))
                        - PvNode;

    bool allowLMR =     depth > 1
                    && !gameCycle
                    && (!kingDangerThem || ss->ply > 6)
                    && (!PvNode || ss->ply > 1);

    bool doLMP =     ss->ply > 2
                 && !PvNode
                 &&  pos.non_pawn_material(us);

    // Step 13. Loop through all pseudo-legal moves until no moves remain
    // or a beta cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (move == excludedMove)
            continue;

        // At root obey the "searchmoves" option and skip moves not listed in Root
        // Move List. In MultiPV mode we also skip PV moves that have been already
        // searched and those of lower "TB rank" if we are in a TB root position.
        if (rootNode && !std::count(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast, move))
            continue;

        ss->moveCount = ++moveCount;

        if (rootNode && is_mainthread() && nodes > NODES_LIMIT_OUTPUT)
        {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
        }
        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension  = 0;
        capture    = pos.capture_stage(move);
        movedPiece = pos.moved_piece(move);
        givesCheck = pos.gives_check(move);
        isMate     = false;

        // This tracks all of our possible responses to our opponent's best moves outside of the PV.
        // The reasoning here is that while we look for flaws in the PV, we must otherwise find an improvement
        // in a secondary root move in order to change the PV. Such an improvement must occur on the path of
        // our opponent's best moves or else it is meaningless.
        ss->secondaryLine = (   (rootNode && moveCount > 1)
                             || (!ourMove && (ss-1)->secondaryLine && !excludedMove && moveCount == 1)
                             || ( ourMove && (ss-1)->secondaryLine));

        ss->mainLine = (   (rootNode && moveCount == 1)
                        || (!ourMove && (ss-1)->mainLine)
                        || ( ourMove && (ss-1)->mainLine && moveCount == 1 && !excludedMove));

        if (givesCheck)
        {
            do_move(pos, move, st, givesCheck, ss);
            isMate = MoveList<LEGAL>(pos).size() == 0;
            undo_move(pos, move);
        }

        if (isMate)
        {
            value = mate_in(ss->ply+1);

            if (PvNode && (moveCount == 1 || value > alpha))
            {
                (ss + 1)->pv = &pv;
                (ss + 1)->pv->clear();
            }
        }
        else
        {
        // Calculate new depth for this move
        newDepth = depth - 1;

        int delta = beta - alpha;

        int r = reduction(improving, depth, moveCount, delta) + ss->ttPv;

        // Step 15. Pruning at shallow depths.
        // Depth conditions are important for mate finding.
        if (    doLMP
            &&  bestValue > VALUE_MATED_IN_MAX_PLY)
        {
            // Skip quiet moves if movecount exceeds our threshold
            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            // SEE based pruning
            if (!pos.see_ge(move, -190 * (depth -1)))
                continue;

            // Reduced depth of the next LMR search
            int lmrDepth = std::max(newDepth - r, newDepth - 4);

            if (   capture
                || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist = captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
                if (!givesCheck && lmrDepth < 7 && !ss->inCheck)
                {
                    Value futilityValue = ss->staticEval + 234 + 247 * lmrDepth
                                        + PieceValue[capturedPiece] + 134 * captHist / 1024;

                    if (futilityValue <= alpha)
                        continue;
                }
            }
            else if (!ss->followPV || !PvNode)
            {
                int dIndex  = std::min(int(depth), int(lmrDivisor.size())) - 1;
                int history = (*contHist[0])[movedPiece][move.to_sq()]
                            + (*contHist[1])[movedPiece][move.to_sq()]
                            + sharedHistory.pawn_entry(pos)[movedPiece][move.to_sq()];

                // Continuation history based pruning
                if (lmrDepth < 6 && history < -4136 * depth)
                    continue;

                history += 69 * mainHistory[us][move.raw()] / 32;

                // (*Scaler): Generally, lower divisors scale well
                lmrDepth += history / lmrDivisor[dIndex];

                Value futilityValue =
                  ss->staticEval + 119 * lmrDepth + 90 * (ss->staticEval > alpha) + 164;

                // Futility pruning: parent node
                // (*Scaler): Generally, more frequent futility pruning
                // scales well
                if (   !ss->inCheck
                    && lmrDepth < (5 * (2 - (ourMove && (ss-1)->secondaryLine)))
                    && history < 20500 - 4097 * (depth - 1)
                    && futilityValue <= alpha)
                    continue;
            }
        }

        // Step 15. Extensions
        if (gameCycleExtension)
            extension = 2;

        // Step 16. Singular Extensions
        //
        // We check for "only moves": if one move fails high on (alpha, beta) but all
        // others fail low on (alpha-s, beta-s), then that move is singular. Singular
        // moves are extended to better estimate the result of the (putatively)
        // forced line. If it's non-singular, we may do some pruning or reduction.
        //
        // Recursive excluded search is avoided. The `excludedMove` mechanism
        // was historically a hack and remains a bit fragile.
        //
        // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
        // and lower extension margins scale well.
        else if (    doSingular
                 &&  move == ttData.move)
        {
            Value singularBeta = std::max(ttData.value - 16 - (1 + (ss->ttPv && !PvNode)) * (depth - 1), -VALUE_MAX_EVAL);
            Depth singularDepth = newDepth / 2;

            ss->excludedMove = move;
            // the search with excludedMove will update ss->staticEval
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();

            if (value < singularBeta)
                extension = allowExt + allowExt * (!PvNode || !ttCapture) * (1 + (value < singularBeta - 224));

            // Multi-cut pruning
            // Our ttMove is assumed to fail high based on the bound of the TT entry,
            // and if after excluding the ttMove with a reduced search we fail high over the original beta,
            // we assume this expected cut-node is not singular (multiple moves fail high),
            // and we can prune the whole subtree by returning a softbound.
            else if (!PvNode && singularBeta >= beta)
                return singularBeta;
        }

        // Step 17. Make the move
        do_move(pos, move, st, givesCheck, ss);

        // Add extension to new depth
        newDepth += extension;

        if (capture)
            ss->statScore = 873 * int(PieceValue[pos.captured_piece()]) / 128
                          + captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())];
        else
            ss->statScore =  2 * mainHistory[us][move.raw()]
                               + (*contHist[0])[movedPiece][move.to_sq()]
                               + (*contHist[1])[movedPiece][move.to_sq()];

        if (move == ttData.move)
            r =   -ss->statScore / 9554;

        else
            r =     r
                  + lmrAdjustment
                  - moveCount / 16
                  - ss->statScore / 9554;

        r -= std::abs(correctionValue) / 26941440;

        if (!capture && !is_decisive(alpha)) // review
            r += std::clamp(alpha - eval, -64, 96) / 343;

        if (!allowExt && r < 0)
            r = 0;

        // Step 17. Late moves reduction / extension (LMR, ~117 Elo)
        // We use various heuristics for the sons of a node after the first son has
        // been searched. In general, we would like to reduce them, but there are many
        // cases where we extend a son if it has good chances to be "interesting".
        if (    allowLMR
            &&  moveCount > 1
            && (!capture || lmrCapture))
        {
            // In general we want to cap the LMR depth search at newDepth, but when
            // reduction is negative, we allow this move a limited search extension
            // beyond the first move depth.
            // To prevent problems when the max value is less than the min value,
            // std::clamp has been replaced by a more robust implementation.
            Depth d = std::max(1, std::min(newDepth - r, newDepth + 2)) + PvNode;

            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);

            // Do a full-depth search when reduced LMR search fails high
            // (*Scaler) Shallower searches here don't scale well
            if (value > alpha)
            {
                // Adjust full-depth search based on LMR results - if the result was
                // good enough search deeper, if it was bad enough search shallower.
                const bool doDeeperSearch = allowExt && d < newDepth && value > bestValue + 53;
                const bool doShallowerSearch = value < bestValue + 8;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                // Post LMR continuation history updates
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1334);
            }
        }

        // Step 19. Full-depth search when LMR is skipped
        else if (!PvNode || moveCount > 1)
        {
            // Increase reduction if ttMove is not present
            if (!ttData.move && abs(beta) < VALUE_MAX_EVAL / 16) //review
                r += 1;

            // If expected reduction is high, we reduce search depth here
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                   newDepth - (r > 5) - (r > 5 && newDepth > 2), !cutNode);
        }

        // Step 20. For PV nodes only, do a full PV search on the first move
        // or after a fail high, otherwise let the parent node fail low with
        // value <= alpha and try another move.
        if (PvNode && (moveCount == 1 || value > alpha))
        {
            (ss + 1)->pv = &pv;
            (ss + 1)->pv->clear();

            // Extend move from transposition table if we are about to dive
            // into qsearch. Decisive score handling improves mate finding
            // and retrograde analysis.
            if (   move == ttData.move
                && allowExt
                // && newDepth < 1
                && ((is_valid(ttData.value) && is_decisive(ttData.value) && ttData.depth > 0)
                    || ttData.depth > 1))
                newDepth = std::max(newDepth, 1);

            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

        // Step 21. Undo move
        undo_move(pos, move);
        }

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 22. Check for a new best move
        // If a stop occurred, the value of the search cannot be trusted,
        // and we return immediately without updating the best move,
        // principal variation or transposition table.
        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (rootNode)
        {
            RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

            if (!is_decisive(value))
                rm.averageScore = rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;
            else
                rm.averageScore = value;

            // PV move or new best move?
            if (moveCount == 1 || value > alpha)
            {
                rm.score = rm.uciScore = value;
                rm.selDepth            = selDepth;
                rm.unset_inexact();

                if (value >= beta)
                {
                    rm.inexactLower = true;
                    rm.uciScore     = beta;
                }
                else if (value <= alpha)
                {
                    rm.inexactUpper = true;
                    rm.uciScore     = alpha;
                }

                rm.pv.resize(1);

                assert((ss + 1)->pv);

                for (Move pvMove : *(ss + 1)->pv)
                    rm.pv.push_back(pvMove);

                // We record how often the best move has been changed in each iteration.
                // This information is used for time management. In MultiPV mode,
                // we must take care to only do this for the first PV line.
                if (moveCount > 1 && !pvIdx)
                    ++bestMoveChanges;
            }
            else
                // All other moves but the PV are set to the lowest value: this
                // is not a problem when sorting because the sort is stable and the
                // move position in the list is preserved -- just the PV is pushed up.
                rm.score = -VALUE_INFINITE;
        }

        // If we have an alternative move equal in value to the current bestmove,
        // we sometimes promote it to bestmove by pretending it just exceeds
        // alpha (but not beta).
        int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                   && !is_decisive(value));

        if (value + inc > bestValue)
        {
            bestValue = value;

            if (value + inc > alpha)
            {
                bestMove = move;

                // Update PV even in fail-high case
                if (PvNode && !rootNode)
                    ss->pv->update(move, (ss + 1)->pv);

                if (value >= beta)
                {
                    // (*Scaler) Infrequent and small updates scale well
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);  // Fail high
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                if (    depth > 3
                    &&  depth < 12
                    && !gameCycle
                    && !is_decisive(value)
                    &&  beta  <  VALUE_MAX_EVAL / 16 // review
                    &&  alpha > -VALUE_MAX_EVAL / 16) // review
                    depth -= 1;

                assert(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta //doublecheck this
            }
        }

        // If the move is worse than some previously searched move,
        // remember it, to update its stats later.
        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    }

    // Step 23. Check for mate and stalemate, otherwise update bestmove/countermove stats

    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    // Adjust best value for fail high cases
    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

    // All legal moves have been searched: if there are no legal moves, it
    // must be a mate or a stalemate (just a fail low score if we are in a
    // singular extension search).
    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : contempt[~us];

    // If there is a move that produces search value greater than alpha,
    // we update the stats of searched moves.
    else if (bestMove)
    {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move, PvNode);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 918 : -747);
    }

    // Bonus for prior quiet countermove that caused the fail low
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = -241;
        bonusScale -= (ss - 1)->statScore / 98;
        bonusScale += std::min(59 * depth, 420);
        bonusScale += 186 * ((ss - 1)->moveCount > 9);
        bonusScale += 142 * (!ss->inCheck && bestValue <= ss->staticEval - 106);
        bonusScale += 159 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 68);

        bonusScale = std::max(bonusScale, 0);

        // scaledBonus ranges from 0 to roughly 2.3M, overflows happen for
        // multipliers larger than 900
        const int scaledBonus = std::min(150 * depth - 85, 1337) * bonusScale;

        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      scaledBonus * 263 / 16384);

        mainHistory[~us][((ss - 1)->currentMove).raw()] << scaledBonus * 215 / 32768;

        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            sharedHistory.pawn_entry(pos)[pos.piece_on(prevSq)][prevSq] << scaledBonus * 324 / 8192;
    }

    // Bonus for prior capture countermove that caused the fail low
    else if (priorCapture && prevSq != SQ_NONE)
    {
        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 892;
    }

    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.
    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

    // Step 24. Write gathered information in transposition table. Note that the
    // static evaluation is saved as it was before correction history.
    if (!excludedMove && !(rootNode && pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                       bestValue >= beta    ? BOUND_LOWER
                       : PvNode && bestMove ? BOUND_EXACT
                                            : BOUND_UPPER,
                       moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                       unadjustedStaticEval, tt.generation(), rule50);

    // Adjust correction history if the best move is not a capture and
    // the error direction matches whether we are above/below bounds.
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && (bestValue > ss->staticEval) == bool(bestMove))
    {
        auto bonus =
          std::clamp(int(bestValue - ss->staticEval) * depth * (bestMove ? 12 : 18) / 128,
                     -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *this, 1061 * bonus / 1024);
    }

    // The search is now complete
    assert(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);
    return bestValue;
}


// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves.
// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search
template<NodeType nodeType>
Value Search::Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {

    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));

    PVMoves   pv;
    StateInfo st;

    Key   posKey;
    Move  move, bestMove;
    Value bestValue, value, futilityBase, drawValue;
    bool  pvHit, givesCheck, capture, gameCycle;
    int   moveCount;
    uint8_t rule50 = std::min(90, pos.rule50_count());
    Color us = pos.side_to_move();

    // Step 1. Initialize node
    if (PvNode)
    {
        (ss + 1)->pv = &pv;
        ss->pv->clear();
    }

    bestMove           = Move::none();
    ss->inCheck        = pos.checkers();
    moveCount          = 0;
    gameCycle          = false;
    drawValue          = contempt[us];

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    if (pos.upcoming_repetition(ss->ply))
    {
       if (drawValue >= beta)
           return drawValue;

       alpha = std::max(alpha, drawValue);
       gameCycle = true;
    }

    if (pos.is_draw(ss->ply))
        return drawValue;

    // Step 2. Check for an immediate draw or maximum ply reached
    if (ss->ply >= MAX_PLY)
        return !ss->inCheck ? evaluate(pos) : drawValue;

    if (alpha >= mate_in(ss->ply+1))
        return mate_in(ss->ply+1);
    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 3. Transposition table lookup
    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);

    ss->ttHit    = ttHit;
    ttData.move  = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply) : VALUE_NONE;
    ttData.value = (abs(ttData.value) > VALUE_MAX_EVAL) ? ttData.value : ttData.value * (100 - rule50) / (100 - ttData.rule50);
    pvHit        = ttHit && ttData.is_pv;

    // At non-PV nodes we check for an early TT cutoff
    if (   !PvNode
        &&  ttData.depth >= DEPTH_QS
        && !gameCycle
        &&  is_valid(ttData.value)
        &&  (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;

    // Step 4. Static evaluation of the position
    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
        bestValue = futilityBase = -VALUE_INFINITE;
    else
    {
        const auto correctionValue = correction_value(*this, pos, ss);

        if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            unadjustedStaticEval = ttData.eval;

            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);

            ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue, rule50);

            // ttValue can be used as a better position evaluation
            if (   !is_decisive(ttData.value)
                &&  ttData.move != Move::none()
                &&  ttData.value > bestValue
                &&  ttData.bound & BOUND_LOWER)
                bestValue = ttData.value;

            else if (   (ss->ply & 1)
                     && ttData.value < bestValue
                     && ttData.bound & BOUND_UPPER)
                bestValue = ttData.value;
        }
        else
        {
            unadjustedStaticEval = evaluate(pos);
            ss->staticEval       = bestValue =
              to_corrected_static_eval(unadjustedStaticEval, correctionValue, rule50);
        }

        // Stand pat. Return immediately if static value is at least beta
        if (bestValue >= beta)
        {
            if (!is_decisive(bestValue))
                bestValue = (441 * bestValue + 583 * beta) / 1024;

            if (!ss->ttHit)
                ttWriter.write(posKey, VALUE_NONE, false, BOUND_LOWER, DEPTH_UNSEARCHED,
                               Move::none(), unadjustedStaticEval, tt.generation(), rule50);

            return bestValue;
        }

        if (bestValue > alpha)
            alpha = bestValue;

        futilityBase = ss->staticEval + 306;
    }

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory};

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // Initialize a MovePicker object for the current position, and prepare
    // to search the moves. We presently use two stages of move generator in
    // quiescence search: captures, or evasions only when in check.
    MovePicker mp(pos, ttData.move, DEPTH_QS, &mainHistory, &lowPlyHistory, &captureHistory,
                  contHist, &sharedHistory, ss->ply);

    // Step 5. Loop through all pseudo-legal moves until no moves remain
    // or a beta cutoff occurs.
    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        moveCount++;

        // Step 6. Pruning
        if (bestValue > VALUE_MATED_IN_MAX_PLY)
        {
            // Futility pruning and moveCount pruning
            if (   !givesCheck
                &&  move.to_sq() != prevSq
                &&  futilityBase > VALUE_MATED_IN_MAX_PLY
                &&  move.type_of() != PROMOTION)
            {
                if (moveCount > 2)
                    continue;

                Value futilityValue = futilityBase + PieceValue[pos.piece_on(move.to_sq())];

                // If static eval + value of piece we are going to capture is
                // much lower than alpha, we can prune this move.
                if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // If static exchange evaluation is low enough, we can prune
                if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::max(bestValue, std::min(alpha, futilityBase));
                    continue;
                }
            }

            // Skip non-captures
            if (   !capture
                && !PvNode)
                continue;

            // Do not search moves with bad enough SEE values
            if (!pos.see_ge(move, -74))
                continue;
        }

        // Step 7. Make and search the move
        do_move(pos, move, st, givesCheck, ss);

        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        undo_move(pos, move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        // Step 8. Check for a new best move
        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                // Update pv even in fail-high case
                if (PvNode)
                    ss->pv->update(move, (ss + 1)->pv);

                if (value < beta) // Update alpha here!
                    alpha = value;
                else
                    break; // Fail high
            }
        }
    }

    // Step 9. Check for mate and stalemate
    // All legal moves have been searched. A special case: if we are
    // in check and no legal moves were found, it is checkmate.
    if (!moveCount)
    {
        if (ss->inCheck)  // Checkmate!
        {
            assert(!MoveList<LEGAL>(pos).size());
            return mated_in(ss->ply);  // Plies to mate from the root
        }

        // Only check for stalemate under specific conditions
        if (!(pawn_single_push_bb(us, pos.pieces(us, PAWN)) & ~pos.pieces())
            && !pos.non_pawn_material(us) && type_of(pos.captured_piece()) >= KNIGHT
            && !MoveList<LEGAL>(pos).size())
            bestValue = VALUE_DRAW;
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (462 * bestValue + 562 * beta) / 1024;

    // Step 10. Save gathered info in transposition table. The static evaluation
    // is saved as it was before adjustment by correction history.
    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation(), rule50);

    // The search is now complete
    assert(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);
    return bestValue;
}

int Search::Worker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return ((reductionScale + 982 - delta * 577 / rootDelta) >> 10) + !i * reductionScale / 2661;
}

// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.
TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}


// Evaluate the current position of the game tree, from the point of view of
// the side to move.
Value Search::Worker::evaluate(const Position& pos) {
    return Eval::evaluate(network[numaAccessToken], pos, accumulatorStack, refreshTable,
                          contempt[pos.side_to_move()]);
}

namespace {

// Adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.
Value value_to_tt(Value v, int ply) {

    assert(v != VALUE_NONE);

    return  v > VALUE_MATE_IN_MAX_PLY   ? v + ply
          : v < VALUE_MATED_IN_MAX_PLY  ? v - ply : v;
  }


  // Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
  // table (which refers to the plies to mate/be mated from current position) to
  // "plies to mate/be mated (TB win/loss) from the root". However, to avoid
  // potentially false mate or TB scores related to the 50 moves rule and the
  // graph history interaction, we return the highest non-TB score instead.
  Value value_from_tt(Value v, int ply) {

    return !is_valid(v)                ? VALUE_NONE
          : v > VALUE_MATE_IN_MAX_PLY  ? v - ply
          : v < VALUE_MATED_IN_MAX_PLY ? v + ply : v;
  }

// Updates stats at the end of search() when a bestMove is found
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Square          prevSq,
                      SearchedList&   quietsSearched,
                      SearchedList&   capturesSearched,
                      Depth           depth,
                      Move            ttMove,
                      bool            PvNode) {

    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

    int bonus =
      std::min(133 * depth - 81, 1487) + 364 * (bestMove == ttMove) + (ss - 1)->statScore / 28;
    int malus = std::min(968 * depth - 235, 2244);

    if (!PvNode)
        // Important: don't remove the cast to a 64-bit number else the multiplication
        // can overflow on 32-bit platforms which would change the bench signature
        bonus += int(bonus * u64(quietsSearched.size() + capturesSearched.size()) / 256);

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 899 / 1024);

        // Decrease stats for all non-best quiet moves
        int actualMalus = malus * 1159 / 1024;
        for (Move move : quietsSearched)
        {
            actualMalus = actualMalus * 921 / 1024;
            update_quiet_histories(pos, ss, workerThread, move, -actualMalus);
        }
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1427 / 1024;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 713 / 1024);

    // Decrease stats for all non-best capture moves
    for (Move move : capturesSearched)
    {
        movedPiece    = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1489 / 1024;
    }
}


// Updates the continuation histories for the move pairs formed by
// the current move and the moves played in previous plies.
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, 520}, {2, 390}, {3, 145}, {4, 251}, {5, 66}, {6, 209}}};

    // Multipliers for positive history consistency
    constexpr int CMHCMultipliers[] = {94, 103, 110, 106, 119, 126, 121};
    int           positiveCount     = 0;

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;

        if (((ss - i)->currentMove).is_ok())
        {
            auto& historyEntry = (*(ss - i)->continuationHistory)[pc][to];
            if (historyEntry > 0)
                positiveCount++;

            int multiplier = CMHCMultipliers[positiveCount];
            historyEntry << bonus * weight * multiplier / 65536 + 73 * (i < 2);
        }
    }
}

// Updates move sorting heuristics

void update_quiet_histories(
  const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.raw()] << bonus;  // Untuned to prevent duplicate effort

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.raw()] << bonus * 712 / 1024;

    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus * 750 / 1024);

    workerThread.sharedHistory.pawn_entry(pos)[pos.moved_piece(move)][move.to_sq()]
      << bonus * (bonus > -4 ? 1104 : 459) / 1024;
}
}


// Function to detect when we are out of available time and stop the search,
// and to print debug info.
void SearchManager::check_time(Search::Worker& worker) {

    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    if (ponder)
        return;

    if ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit))
        || (worker.limits.movetime && elapsed >= worker.limits.movetime)
        || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes))
        worker.threads.stop = true;
}


void SearchManager::output_pv(Search::Worker&           worker,
                              const ThreadPool&         threads,
                              const TranspositionTable& tt,
                              Depth                     depth) {

    const auto nodes     = threads.nodes_searched();
    const auto contempt  = UCIEngine::to_int(int(worker.options["Contempt"]), worker.rootPos);
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    usize      multiPV   = std::min(usize(worker.options["MultiPV"]), rootMoves.size());
    u64        tbHits    = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);

    for (usize i = 0; i < multiPV; ++i)
    {
        bool usePreviousScore = rootMoves[i].score == -VALUE_INFINITE;

        if (depth == 1 && usePreviousScore && i > 0)
            continue;

        Depth d = usePreviousScore ? std::max(1, depth - 1) : depth;
        Value v = usePreviousScore ? rootMoves[i].previousScore : rootMoves[i].uciScore;

        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;

        bool isTBScore = worker.tbConfig.rootInTB && !is_mate_or_mated(v); // was !is_decisive(v)
        v              = isTBScore ? rootMoves[i].tbScore : v;

        if (contempt > 0 && !is_decisive(v))
        {
            if (v >= contempt)
                v -= contempt;

            else if (v <= -contempt)
                v += contempt;
        }

        std::string pv;
        for (Move m : usePreviousScore ? rootMoves[i].previousPV : rootMoves[i].pv)
            pv += UCIEngine::move(m, pos.is_chess960()) + " ";

        // Remove last whitespace
        if (!pv.empty())
            pv.pop_back();

        auto wdl = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";

        // Scores cannot be both exact and inexact
        assert(!(rootMoves[i].inexactLower && rootMoves[i].inexactUpper));
        auto bound = rootMoves[i].inexactLower ? "lowerbound"
                   : rootMoves[i].inexactUpper ? "upperbound"
                                               : "";

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
        info.score    = {v, pos};
        info.wdl      = wdl;

        // TB and previous scores are exact, even though their flags may say otherwise
        if (!(isTBScore || usePreviousScore))
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
        info.tbHits    = tbHits;
        info.pv        = pv;
        info.hashfull  = tt.hashfull();

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI, otherwise
// in case of 'ponder on' we have nothing to think about.
bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {

    assert(pv.size() == 1 && pv[0] != Move::none());

    StateInfo st;
    pos.do_move(pv[0], st, &tt);

    if (!pos.is_draw(1))
    {
        auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
        if (ttHit && MoveList<LEGAL>(pos).contains(ttData.move))
            pv.push_back(ttData.move);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}


}  // namespace Stockfish
