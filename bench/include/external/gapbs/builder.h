// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

/**
 * @file builder.h
 * @brief Graph construction and reordering framework for GAP Benchmark Suite
 * 
 * This file provides the BuilderBase class which handles:
 * - Graph construction from edge lists (file or synthetic)
 * - Graph reordering using various algorithms
 * - CSR format manipulation and transformations
 * 
 * ARCHITECTURE:
 * - Core graph operations: MakeGraph(), MakeGraphFromEL(), SquishCSR()
 * - Reordering dispatch: GenerateMapping() - main entry point
 * - Algorithm delegates: Each Generate*Mapping() calls reorder/*.h implementations
 * 
 * REORDERING ALGORITHMS (see reorder/*.h for implementations):
 * - Basic (0-2): Original, Random, Sort
 * - Hub (3-7): HubSort, HubCluster, DBG variants
 * - RabbitOrder (8): Community-aware hierarchical clustering
 * - Classic (9-11): GOrder, COrder, RCMOrder
 * - GraphBrew (12): Multi-level community-based reordering
 * - Adaptive (14): ML-based per-community algorithm selection
 * - Leiden (15): GVE-Leiden baseline (community detection reference)
 * 
 * Author: Scott Beamer (original), GraphBrew Team (extensions)
 */

// ============================================================================
// STANDARD LIBRARY INCLUDES
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================================
// LOCAL INCLUDES
// ============================================================================

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "sliding_queue.h"
#include "timer.h"
#include "util.h"

// Unified reorder header - provides all algorithm implementations
#include <reorder/reorder.h>

// Cagra/GraphIT partitioning helpers (P-OPT)
#include <partition/cagra/popt.h>

// Graph partitioning (TRUST algorithm)
#include <partition/trust.h>

// ============================================================================
// EXTERNAL LIBRARY INCLUDES
// ============================================================================

#ifdef RABBIT_ENABLE
#include <rabbit/edge_list.hpp>
#include <rabbit/rabbit_order.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>
using namespace edge_list;
#endif

// GOrder includes (MIT License - Hao Wei, 2016)
#include <gorder/GoGraph.h>
#include <gorder/GoUtil.h>

/*
 * @author Priyank Faldu <Priyank.Faldu@ed.ac.uk> <http://faldupriyank.com>
 *
 * Copyright 2019 The University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <corder/vec2d.h>

#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 1
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 16
#endif

// ============================================================================
// UNIFIED LEIDEN DEFAULTS - For fair comparison across all Leiden algorithms
// ============================================================================
#ifndef LEIDEN_DEFAULT_ITERATIONS
/** Number of iterations per Leiden pass - controls local move refinement */
#define LEIDEN_DEFAULT_ITERATIONS 20
#endif
#ifndef LEIDEN_DEFAULT_PASSES
/** Number of Leiden passes - controls coarsening depth */
#define LEIDEN_DEFAULT_PASSES 10
#endif

#include <leiden/main.hxx>

enum class GraphInputPolicy
{
    kDefault,
    kPreserveDirectedEdges,
};

template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_, bool invert = true,
          typename FNodeID_ = NodeID_, typename FDestID_ = NodeID_>
class BuilderBase
{
    typedef EdgePair<NodeID_, DestID_> Edge;
    typedef pvector<Edge> EdgeList;

    const CLBase &cli_;
    bool symmetrize_;
    bool needs_weights_;
    bool in_place_ = false;
    int64_t num_nodes_ = -1;
    std::vector<std::pair<ReorderingAlgo, std::string>> reorder_options_;

    void ValidatePreservedInputOptions() const
    {
        if (symmetrize_)
            throw std::invalid_argument(
                "preserved directed-edge input cannot be symmetrized");
        if (in_place_)
            throw std::invalid_argument(
                "preserved directed-edge input cannot use in-place building");
        if (!cli_.reorder_options().empty())
            throw std::invalid_argument(
                "preserved directed-edge input cannot be reordered");
    }

public:
    explicit BuilderBase(const CLBase &cli) : cli_(cli)
    {

        symmetrize_ = cli_.symmetrize();
        needs_weights_ = !std::is_same<NodeID_, DestID_>::value;
        in_place_ = cli_.in_place();
        // reorder_options_(cli_.reorder_options());
        if (in_place_ && needs_weights_)
        {
            std::cout << "In-place building (-m) does not support weighted graphs"
                      << std::endl;
            exit(-30);
        }
    }

    DestID_ GetSource(EdgePair<NodeID_, NodeID_> e)
    {
        return e.u;
    }

    DestID_ GetSource(EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_>> e)
    {
        return NodeWeight<NodeID_, WeightT_>(e.u, e.v.w);
    }

    NodeID_ FindMaxNodeID(const EdgeList &el)
    {
        NodeID_ max_seen = 0;
        #pragma omp parallel for reduction(max : max_seen)
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            max_seen = __gnu_parallel::max(max_seen, e.u);
            max_seen = __gnu_parallel::max(max_seen, (NodeID_)e.v);
        }
        return max_seen;
    }

    NodeID_ FindMinNodeID(const EdgeList &el)
    {
        NodeID_ min_seen = FindMaxNodeID(el);
        #pragma omp parallel for reduction(min : min_seen)
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            min_seen = __gnu_parallel::min(min_seen, e.u);
            min_seen = __gnu_parallel::min(min_seen, (NodeID_)e.v);
        }
        return min_seen;
    }

    pvector<NodeID_> CountDegrees(const EdgeList &el, bool transpose)
    {
        pvector<NodeID_> degrees(num_nodes_, 0);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if (symmetrize_ || (!symmetrize_ && !transpose))
                fetch_and_add(degrees[e.u], 1);
            if ((symmetrize_ && !in_place_) || (!symmetrize_ && transpose))
                fetch_and_add(degrees[(NodeID_)e.v], 1);
        }
        return degrees;
    }

    static pvector<SGOffset> PrefixSum(const pvector<NodeID_> &degrees)
    {
        pvector<SGOffset> sums(degrees.size() + 1);
        SGOffset total = 0;
        for (size_t n = 0; n < degrees.size(); n++)
        {
            sums[n] = total;
            total += degrees[n];
        }
        sums[degrees.size()] = total;
        return sums;
    }

    static pvector<SGOffset> ParallelPrefixSum(const pvector<NodeID_> &degrees)
    {
        const size_t block_size = 1 << 20;
        const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
        pvector<SGOffset> local_sums(num_blocks);
        #pragma omp parallel for
        for (size_t block = 0; block < num_blocks; block++)
        {
            SGOffset lsum = 0;
            size_t block_end = std::min((block + 1) * block_size, degrees.size());
            for (size_t i = block * block_size; i < block_end; i++)
                lsum += degrees[i];
            local_sums[block] = lsum;
        }
        pvector<SGOffset> bulk_prefix(num_blocks + 1);
        SGOffset total = 0;
        for (size_t block = 0; block < num_blocks; block++)
        {
            bulk_prefix[block] = total;
            total += local_sums[block];
        }
        bulk_prefix[num_blocks] = total;
        pvector<SGOffset> prefix(degrees.size() + 1);
        #pragma omp parallel for
        for (size_t block = 0; block < num_blocks; block++)
        {
            SGOffset local_total = bulk_prefix[block];
            size_t block_end = std::min((block + 1) * block_size, degrees.size());
            for (size_t i = block * block_size; i < block_end; i++)
            {
                prefix[i] = local_total;
                local_total += degrees[i];
            }
        }
        prefix[degrees.size()] = bulk_prefix[num_blocks];
        return prefix;
    }

    // Removes self-loops and redundant edges
    // Side effect: neighbor IDs will be sorted
    void SquishCSR(const CSRGraph<NodeID_, DestID_, invert> &g, bool transpose,
                   DestID_ ***sq_index, DestID_ **sq_neighs)
    {
        pvector<NodeID_> diffs(g.num_nodes());
        DestID_ *n_start, *n_end;
        #pragma omp parallel for private(n_start, n_end)
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            if (transpose)
            {
                n_start = g.in_neigh(n).begin();
                n_end = g.in_neigh(n).end();
            }
            else
            {
                n_start = g.out_neigh(n).begin();
                n_end = g.out_neigh(n).end();
            }
            __gnu_parallel::stable_sort(n_start, n_end);
            DestID_ *new_end = std::unique(n_start, n_end);
            if(!cli_.keep_self())
                new_end = std::remove(n_start, new_end, n);
            diffs[n] = new_end - n_start;
        }
        pvector<SGOffset> sq_offsets = ParallelPrefixSum(diffs);
        *sq_neighs = new DestID_[sq_offsets[g.num_nodes()]];
        *sq_index = CSRGraph<NodeID_, DestID_>::GenIndex(sq_offsets, *sq_neighs);
        #pragma omp parallel for private(n_start)
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            if (transpose)
                n_start = g.in_neigh(n).begin();
            else
                n_start = g.out_neigh(n).begin();
            std::copy(n_start, n_start + diffs[n], (*sq_index)[n]);
        }
    }

    CSRGraph<NodeID_, DestID_, invert>
    SquishGraph(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        DestID_ **out_index, *out_neighs, **in_index, *in_neighs;
        SquishCSR(g, false, &out_index, &out_neighs);
        if (g.directed())
        {
            if (invert)
                SquishCSR(g, true, &in_index, &in_neighs);
            return CSRGraph<NodeID_, DestID_, invert>(
                       g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
        }
        else
        {
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                    out_neighs);
        }
    }

    /*
       In-Place Graph Building Steps
       - sort edges and squish (remove self loops and redundant edges)
       - overwrite EdgeList's memory with outgoing neighbors
       - if graph not being symmetrized
        - finalize structures and make incoming structures if requested
       - if being symmetrized
        - search for needed inverses, make room for them, add them in place
     */
    void MakeCSRInPlace(EdgeList &el, DestID_ ***index, DestID_ **neighs,
                        DestID_ ***inv_index, DestID_ **inv_neighs)
    {
        // preprocess EdgeList - sort & squish in place
        __gnu_parallel::stable_sort(el.begin(), el.end());
        auto new_end = std::unique(el.begin(), el.end());
        el.resize(new_end - el.begin());
        auto self_loop = [](Edge e)
        {
            return e.u == e.v;
        };
        new_end = std::remove_if(el.begin(), el.end(), self_loop);
        el.resize(new_end - el.begin());
        // analyze EdgeList and repurpose it for outgoing edges
        pvector<NodeID_> degrees = CountDegrees(el, false);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        pvector<NodeID_> indegrees = CountDegrees(el, true);
        *neighs = reinterpret_cast<DestID_ *>(el.data());
        for (Edge e : el)
            (*neighs)[offsets[e.u]++] = e.v;
        size_t num_edges = el.size();
        el.leak();
        // revert offsets by shifting them down
        for (NodeID_ n = num_nodes_; n >= 0; n--)
            offsets[n] = n != 0 ? offsets[n - 1] : 0;
        if (!symmetrize_) // not going to symmetrize so no need to add edges
        {
            size_t new_size = num_edges * sizeof(DestID_);
            *neighs = static_cast<DestID_ *>(std::realloc(*neighs, new_size));
            *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
            if (invert) // create inv_neighs & inv_index for incoming edges
            {
                pvector<SGOffset> inoffsets = ParallelPrefixSum(indegrees);
                *inv_neighs = new DestID_[inoffsets[num_nodes_]];
                *inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inoffsets, *inv_neighs);
                for (NodeID_ u = 0; u < num_nodes_; u++)
                {
                    for (DestID_ *it = (*index)[u]; it < (*index)[u + 1]; it++)
                    {
                        NodeID_ v = static_cast<NodeID_>(*it);
                        (*inv_neighs)[inoffsets[v]] = u;
                        inoffsets[v]++;
                    }
                }
            }
        }
        else   // symmetrize graph by adding missing inverse edges
        {
            // Step 1 - count number of needed inverses
            pvector<NodeID_> invs_needed(num_nodes_, 0);
            for (NodeID_ u = 0; u < num_nodes_; u++)
            {
                for (SGOffset i = offsets[u]; i < offsets[u + 1]; i++)
                {
                    DestID_ v = (*neighs)[i];
                    bool inv_found =
                        std::binary_search(*neighs + offsets[v], *neighs + offsets[v + 1],
                                           static_cast<DestID_>(u));
                    if (!inv_found)
                        invs_needed[v]++;
                }
            }
            // increase offsets to account for missing inverses, realloc neighs
            SGOffset total_missing_inv = 0;
            for (NodeID_ n = 0; n < num_nodes_; n++)
            {
                offsets[n] += total_missing_inv;
                total_missing_inv += invs_needed[n];
            }
            offsets[num_nodes_] += total_missing_inv;
            size_t newsize = (offsets[num_nodes_] * sizeof(DestID_));
            *neighs = static_cast<DestID_ *>(std::realloc(*neighs, newsize));
            if (*neighs == nullptr)
            {
                std::cout << "Call to realloc() failed" << std::endl;
                exit(-33);
            }
            // Step 2 - spread out existing neighs to make room for inverses
            //   copies backwards (overwrites) and inserts free space at starts
            SGOffset tail_index = offsets[num_nodes_] - 1;
            for (NodeID_ n = num_nodes_ - 1; n >= 0; n--)
            {
                SGOffset new_start = offsets[n] + invs_needed[n];
                for (SGOffset i = offsets[n + 1] - 1; i >= new_start; i--)
                {
                    (*neighs)[tail_index] = (*neighs)[i - total_missing_inv];
                    tail_index--;
                }
                total_missing_inv -= invs_needed[n];
                tail_index -= invs_needed[n];
            }
            // Step 3 - add missing inverse edges into free spaces from Step 2
            for (NodeID_ u = 0; u < num_nodes_; u++)
            {
                for (SGOffset i = offsets[u] + invs_needed[u]; i < offsets[u + 1];
                        i++)
                {
                    DestID_ v = (*neighs)[i];
                    bool inv_found = std::binary_search(
                                         *neighs + offsets[v] + invs_needed[v], *neighs + offsets[v + 1],
                                         static_cast<DestID_>(u));
                    if (!inv_found)
                    {
                        (*neighs)[offsets[v] + invs_needed[v] - 1] =
                            static_cast<DestID_>(u);
                        invs_needed[v]--;
                    }
                }
            }
            for (NodeID_ n = 0; n < num_nodes_; n++)
                __gnu_parallel::stable_sort(*neighs + offsets[n],
                                            *neighs + offsets[n + 1]);
            *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
        }
    }

    /*
       Graph Building Steps (for CSR):
       - Read edgelist once to determine vertex degrees (CountDegrees)
       - Determine vertex offsets by a prefix sum (ParallelPrefixSum)
       - Allocate storage and set points according to offsets (GenIndex)
       - Copy edges into storage
     */
    void MakeCSR(const EdgeList &el, bool transpose, DestID_ ***index,
                 DestID_ **neighs)
    {
        pvector<NodeID_> degrees = CountDegrees(el, transpose);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        *neighs = new DestID_[offsets[num_nodes_]];
        *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if (symmetrize_ || (!symmetrize_ && !transpose))
                (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
            if (symmetrize_ || (!symmetrize_ && transpose))
                (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
                    GetSource(e);
        }
    }

    CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList &el)
    {
        DestID_ **index = nullptr, **inv_index = nullptr;
        DestID_ *neighs = nullptr, *inv_neighs = nullptr;
        Timer t;
        t.Start();
        if (num_nodes_ == -1)
            num_nodes_ = FindMaxNodeID(el) + 1;
        if (needs_weights_)
            Generator<NodeID_, DestID_, WeightT_>::InsertWeights(el);
        if (in_place_)
        {
            MakeCSRInPlace(el, &index, &neighs, &inv_index, &inv_neighs);
        }
        else
        {
            MakeCSR(el, false, &index, &neighs);
            if (!symmetrize_ && invert)
            {
                MakeCSR(el, true, &inv_index, &inv_neighs);
            }
        }
        t.Stop();

        PrintTime("Build Time", t.Seconds());
        if (symmetrize_)
            return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs);
        else
            return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs,
                    inv_index, inv_neighs);
    }

    CSRGraph<NodeID_, DestID_, invert> MakeGraphFromELPreservingInput(
        EdgeList &el,
        int64_t explicit_num_nodes = -1)
    {
        ValidatePreservedInputOptions();
        if (explicit_num_nodes < -1)
            throw std::invalid_argument("explicit vertex count cannot be negative");
        if (explicit_num_nodes == -1 && el.empty())
            throw std::invalid_argument(
                "empty edge-list input has no explicit vertex count; "
                "use .sg/.wsg or a sized .mtx/.graph input");
        const uint64_t node_limit = std::min<uint64_t>(
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<NodeID_>::max());
        uint64_t inferred_nodes = 0;
        for (const Edge &edge : el)
        {
            const NodeID_ source = edge.u;
            const NodeID_ destination = static_cast<NodeID_>(edge.v);
            if constexpr (std::is_signed<NodeID_>::value)
                if (source < 0 || destination < 0)
                    throw std::invalid_argument("preserved input has a negative vertex");
            const uint64_t highest = std::max<uint64_t>(source, destination);
            if (highest >= node_limit ||
                (explicit_num_nodes >= 0 &&
                 highest >= static_cast<uint64_t>(explicit_num_nodes)))
                throw std::invalid_argument("preserved edge exceeds the vertex domain");
            inferred_nodes = std::max(inferred_nodes, highest + 1);
        }
        if (explicit_num_nodes >= 0 &&
            static_cast<uint64_t>(explicit_num_nodes) > node_limit)
            throw std::invalid_argument("explicit vertex count exceeds the index type");
        num_nodes_ = explicit_num_nodes >= 0
            ? explicit_num_nodes : static_cast<int64_t>(inferred_nodes);
        return MakeGraphFromEL(el);
    }

    pvector<NodeID_> CountLocalDegrees(const EdgeList &el, bool transpose, int64_t num_nodes_local = -1)
    {
        pvector<NodeID_> degrees(num_nodes_local, 0);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if ((!transpose))
                fetch_and_add(degrees[e.u], 1);
            if ((transpose))
                fetch_and_add(degrees[(NodeID_)e.v], 1);
        }
        return degrees;
    }

    void MakeLocalCSR(const EdgeList &el, bool transpose, DestID_ ***index,
                      DestID_ **neighs)
    {
        int64_t num_nodes_local = FindMaxNodeID(el) + 1;
        pvector<NodeID_> degrees = CountLocalDegrees(el, transpose, num_nodes_local);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        *neighs = new DestID_[offsets[num_nodes_local]];
        *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if ((!transpose))
                (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
            if ((transpose))
                (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
                    GetSource(e);
        }
    }

    CSRGraph<NodeID_, DestID_, invert> MakeLocalGraphFromEL(EdgeList &el, bool verbose = false)
    {
        DestID_ **index = nullptr;
        // **inv_index = nullptr;
        DestID_ *neighs = nullptr;
        // *inv_neighs = nullptr;
        Timer t;
        t.Start();

        int64_t num_nodes_local = FindMaxNodeID(el) + 1;

        MakeLocalCSR(el, false, &index, &neighs);
        // MakeLocalCSR(el, true, &inv_index, &inv_neighs);
        // CSRGraph<NodeID_, DestID_, invert> g = CSRGraph<NodeID_, DestID_, invert>(
        //         num_nodes_, index, neighs, inv_index, inv_neighs);
        CSRGraph<NodeID_, DestID_, invert> g = CSRGraph<NodeID_, DestID_,
                                           invert>(num_nodes_local, index, neighs);

        g = SquishGraph(g);
        // SquishCSR(g, false, &index, &neighs);
        // SquishCSR(g, true, &inv_index, &inv_neighs);
        t.Stop();
        if (verbose) {
            PrintTime("Local Build Time", t.Seconds());
        }
        return g;
        // return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }

    void FlattenPartitions(
        const std::vector<CSRGraph<NodeID_, DestID_, invert>> &partitions,
        std::vector<CSRGraphFlat<FNodeID_, WeightT_, FDestID_>> &partitions_flat,
        size_t alignment = 4096)
    {
        partitions_flat.reserve(
            partitions.size()); // Reserve space for the flattened partitions

        for (const auto &partition : partitions)
        {
            partitions_flat.push_back(partition.flattenGraphOut(alignment));
        }
    }

    void
    MakeOrientedELFromUniDirect(EdgeList &el,
                                const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        int64_t num_edges = el.size();

        // Parallel loop to TRUST oriented edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_edges; ++i)
        {
            Edge e = el[i];
            if (g.is_weighted())
            {
                NodeID_ src = e.u;
                NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(e.v).v;
                WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(e.v).w;

                if (src > dest)
                    el[i] = Edge(dest, NodeWeight<NodeID_, WeightT_>(src, weight));

            }
            else
            {
                NodeID_ src = e.u;
                NodeID_ dest = e.v;

                if (src > dest)
                    el[i] = Edge(dest, src);
            }
        }
    }

    EdgeList
    MakeUniDirectELFromGraph(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ min_seen = 0)
    {
        int64_t num_edges = g.num_edges_directed();
        int64_t num_nodes = g.num_nodes();
        EdgeList el(num_edges * 2);
        el.resize(num_edges * 2);

        // Parallel loop to construct the edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            NodeID_ out_start = g.out_offset(i);
            NodeID_ in_start = out_start + num_edges;

            NodeID_ j = 0;
            for (DestID_ neighbor : g.out_neigh(i))
            {
                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v - min_seen;
                    WeightT_ weight =
                        static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    el[out_start + j] = Edge(i - min_seen, NodeWeight<NodeID_, WeightT_>(dest, weight));
                    el[in_start + j] =
                        Edge(dest, NodeWeight<NodeID_, WeightT_>(i - min_seen, weight));
                }
                else
                {
                    el[out_start + j] = Edge(i - min_seen, neighbor - min_seen);
                    el[in_start + j] = Edge(neighbor - min_seen, i - min_seen);
                }

                ++j;
            }
        }

        // PrintEdgeList(el);

        return el;
    }

    void PrintEdgeList(const EdgeList &el)
    {
        for (const auto &edge : el)
        {
            std::cout << edge.u << " -> " << edge.v << std::endl;
        }
    }

    /**
     * @brief Partitions a graph based on input parameters and returns the
     * partitions.
     *
     * This function retrieves partitioning parameters from the CLI input, creates
     * a graph, and then partitions the graph using either the GRAPHIT/Cagra or
     * TRUST partitioning method based on the input type.
     *
     * For TRUST partitioning, uses TrustPartitioner from partition/trust.h
     * For Cagra partitioning, uses MakeCagraPartitionedGraph from cache/popt.h
     *
     * @return std::vector<CSRGraph<NodeID_, DestID_, invert>> The partitions of
     * the graph.
     */
    std::vector<CSRGraph<NodeID_, DestID_, invert>> MakePartitionedGraph()
    {
        std::vector<CSRGraph<NodeID_, DestID_, invert>> partitions;

        std::vector<int>::const_iterator segment_iter = cli_.segments().begin();
        int p_type = *segment_iter;
        segment_iter++;
        int p_n = *segment_iter;
        segment_iter++;
        int p_m = *segment_iter;
        segment_iter++;

        CSRGraph<NodeID_, DestID_, invert> g = MakeGraph();

        switch (p_type)
        {
        case 0: // <0:GRAPHIT/Cagra>
            partitions = ::MakeCagraPartitionedGraph<NodeID_, DestID_, invert>(
                g, p_n, p_m, cli_.use_out_degree());
            break;
        case 1: // <1:TRUST>
            {
                // Use standalone TrustPartitioner from partition/trust.h
                TrustPartitioner<NodeID_, DestID_, WeightT_, invert> trustPartitioner(cli_.logging_en());
                partitions = trustPartitioner.MakeTrustPartitionedGraph(g, p_n, p_m);
            }
            break;
        default:
            partitions = ::MakeCagraPartitionedGraph<NodeID_, DestID_, invert>(
                g, p_n, p_m, cli_.use_out_degree());
            break;
        }

        return partitions;
    }

    void PrintPartitionsTopology(
        const std::vector<CSRGraph<NodeID_, DestID_, invert>> &partitions)
    {
        for (size_t i = 0; i < partitions.size(); ++i)
        {
            std::cout << "Partition " << i << ":" << std::endl;
            partitions[i].PrintTopology();
        }
    }

    CSRGraph<NodeID_, DestID_, invert> MakeGraph(
        GraphInputPolicy input_policy = GraphInputPolicy::kDefault)
    {
        const bool preserve_input_edges =
            input_policy == GraphInputPolicy::kPreserveDirectedEdges;
        if (preserve_input_edges && cli_.filename().empty())
            throw std::invalid_argument(
                "preserved directed-edge input requires an explicit graph file");
        if (preserve_input_edges)
            ValidatePreservedInputOptions();
        CSRGraph<NodeID_, DestID_, invert> g;
        CSRGraph<NodeID_, DestID_, invert> g_final;
        bool gContinue_ = true; // Control variable to exit the scope
        {
            // extra scope to trigger earlier deletion of el (save memory)
            EdgeList el;
            if (cli_.filename() != "")
            {
                Reader<NodeID_, DestID_, WeightT_, invert> r(cli_.filename());
                if ((r.GetSuffix() == ".sg") || (r.GetSuffix() == ".wsg"))
                {
                    g_final = r.ReadSerializedGraph();
                    gContinue_ = false; // Control variable to exit the scope
                }
                else
                {
                    el = r.ReadFile(needs_weights_);
                    if (preserve_input_edges && r.explicit_num_nodes() >= 0)
                        num_nodes_ = r.explicit_num_nodes();
                }
            }
            else if (cli_.scale() != -1)
            {
                Generator<NodeID_, DestID_> gen(cli_.scale(), cli_.degree());
                el = gen.GenerateEL(cli_.uniform());
            }
            if (gContinue_)
            {
                g = preserve_input_edges
                    ? MakeGraphFromELPreservingInput(el, num_nodes_)
                    : MakeGraphFromEL(el);
            }
        }

        if (gContinue_)
        {
            if (in_place_ || preserve_input_edges)
                g_final = std::move(g);
            else
                g_final = SquishGraph(g);
        }
        
        // Auto-set graph name hint from input filename for database/oracle modes
        if (!cli_.filename().empty()) {
            SetGraphNameHint(ExtractGraphNameFromPath(cli_.filename()));
        }
        
        // Compute and print global graph topology features ONCE before any reordering
        // This provides clustering_coeff, avg_path_length, diameter, community_count
        // for the Python training scripts to use
        ComputeAndPrintGlobalTopologyFeatures(g_final);
        
        // g_final.PrintTopology();
        pvector<NodeID_> new_ids(g_final.num_nodes(), -1);
        for (const auto &option : cli_.reorder_options())
        {
            new_ids.fill(-1);
            GenerateMapping(g_final, new_ids, option.first, cli_.use_out_degree(),
                            option.second);

            // Debug-mode bijection check: verify mapping is a valid permutation
            #ifndef NDEBUG
            {
                std::vector<bool> seen(g_final.num_nodes(), false);
                for (NodeID_ v = 0; v < g_final.num_nodes(); ++v) {
                    assert(new_ids[v] >= 0 && new_ids[v] < g_final.num_nodes()
                           && "Mapping ID out of range");
                    assert(!seen[new_ids[v]]
                           && "Duplicate mapping ID — not a permutation");
                    seen[new_ids[v]] = true;
                }
            }
            #endif

            g_final = RelabelByMapping(g_final, new_ids);
        }

        // g_final = SquishGraph(g_final);
        // g_final.PrintTopology();
        // g_final.PrintTopologyOriginal();
        return g_final;
    }

    // Relabels (and rebuilds) graph by order of decreasing degree
    static CSRGraph<NodeID_, DestID_, invert>
    RelabelByDegree(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        if (g.directed())
        {
            std::cout << "Cannot relabel directed graph" << std::endl;
            std::exit(-11);
        }
        Timer t;
        t.Start();
        typedef std::pair<int64_t, NodeID_> degree_node_p;
        pvector<degree_node_p> degree_id_pairs(g.num_nodes());
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
            degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
        __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                                    std::greater<degree_node_p>());
        pvector<NodeID_> degrees(g.num_nodes());
        pvector<NodeID_> new_ids(g.num_nodes());
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            degrees[n] = degree_id_pairs[n].first;
            new_ids[degree_id_pairs[n].second] = n;
        }
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        #pragma omp parallel for
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            for (NodeID_ v : g.out_neigh(u))
                neighs[offsets[new_ids[u]]++] = new_ids[v];
            std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
        }
        t.Stop();
        PrintTime("Relabel", t.Seconds());
        return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }

    static CSRGraph<NodeID_, DestID_, invert>
    RelabelByMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                     pvector<NodeID_> &new_ids)
    {
        Timer t;
        t.Start();
        bool outDegree = true;
        // bool createOnlyDegList = true;
        CSRGraph<NodeID_, DestID_, invert> g_relabel;
        bool createBothCSRs = true;

        auto max_iter = __gnu_parallel::max_element(new_ids.begin(), new_ids.end());
        size_t max_id = *max_iter;

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            if (new_ids[v] == -1)
            {
                // Assigning new IDs starting from max_id atomically
                NodeID_ local_max = __sync_fetch_and_add(&max_id, 1);
                new_ids[v] = local_max + 1;
                // cerr << v << " " << new_ids[v] << " " << max_id << endl;
            }
        }

        if (g.directed() == true)
        {
            #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
                assert(new_ids[v] != -1);

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            pvector<NodeID_> inv_degrees(g.num_nodes());
            if (outDegree == true)
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.out_degree(n);
                    inv_degrees[new_ids[n]] = g.in_degree(n);
                }
            }
            else
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.in_degree(n);
                    inv_degrees[new_ids[n]] = g.out_degree(n);
                }
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(inv_degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                if (outDegree == true)
                {
                    for (NodeID_ v : g.in_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                else
                {
                    for (NodeID_ v : g.out_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                std::sort(index[new_ids[u]],
                          index[new_ids[u] + 1]); // sort neighbors of each vertex
            }
            DestID_ *inv_neighs(nullptr);
            DestID_ **inv_index(nullptr);
            if (createBothCSRs == true)
            {
                // making the inverse list (in-degrees in this case)
                pvector<SGOffset> inv_offsets = ParallelPrefixSum(degrees);
                inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
                inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
                if (createBothCSRs == true)
                {
                    #pragma omp parallel for schedule(dynamic, 1024)
                    for (NodeID_ u = 0; u < g.num_nodes(); u++)
                    {
                        if (outDegree == true)
                        {
                            for (NodeID_ v : g.out_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        else
                        {
                            for (NodeID_ v : g.in_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        std::sort(
                            inv_index[new_ids[u]],
                            inv_index[new_ids[u] + 1]); // sort neighbors of each vertex
                    }
                }
            }
            t.Stop();
            PrintTime("Relabel Map Time", t.Seconds());
            if (outDegree == true)
            {

                g_relabel = CSRGraph<NodeID_, DestID_, invert>(
                                g.num_nodes(), inv_index, inv_neighs, index, neighs);
            }
            else
            {
                g_relabel = CSRGraph<NodeID_, DestID_, invert>(
                                g.num_nodes(), index, neighs, inv_index, inv_neighs);
            }
        }
        else
        {
            /* Undirected graphs - no need to make separate lists for in and out
             * degree */

            #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
                assert(new_ids[v] != -1);

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                degrees[new_ids[n]] = g.out_degree(n);
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                for (NodeID_ v : g.out_neigh(u))
                    neighs[offsets[new_ids[u]]++] = new_ids[v];
                std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
            }
            t.Stop();
            PrintTime("Relabel Map Time", t.Seconds());
            g_relabel =
                CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
        }

        g_relabel.copy_org_ids(g.get_org_ids());
        g_relabel.update_org_ids(new_ids);
        return g_relabel;
    }

    static CSRGraph<NodeID_, DestID_, invert>
    RelabelByMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                        pvector<NodeID_> &new_ids)
    {
        Timer t;
        DestID_ **out_index;
        DestID_ *out_neighs;
        DestID_ **in_index;
        DestID_ *in_neighs;
        CSRGraph<NodeID_, DestID_, invert> g_relabel;

        t.Start();
        pvector<NodeID_> out_degrees(g.num_nodes(), 0);

        auto max_iter = __gnu_parallel::max_element(new_ids.begin(), new_ids.end());
        size_t max_id = *max_iter;

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            if (new_ids[v] == -1)
            {
                // Assigning new IDs starting from max_id atomically
                NodeID_ local_max = __sync_fetch_and_add(&max_id, 1);
                new_ids[v] = local_max + 1;
            }
        }

        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            out_degrees[new_ids[n]] = g.out_degree(n);
            // if(new_ids[n] > g.num_nodes())
            // cerr << new_ids[n] << endl;
        }
        pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
        out_neighs = new DestID_[out_offsets[g.num_nodes()]];
        out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
        #pragma omp parallel for
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            for (NodeID_ v : g.out_neigh(u))
            {
                SGOffset out_offsets_local =
                    __sync_fetch_and_add(&(out_offsets[new_ids[u]]), 1);
                out_neighs[out_offsets_local] = new_ids[v];
            }
            std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
        }

        if (g.directed())
        {
            pvector<NodeID_> in_degrees(g.num_nodes(), 0);
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                in_degrees[new_ids[n]] = g.in_degree(n);
            }
            pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
            in_neighs = new DestID_[in_offsets[g.num_nodes()]];
            in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
            #pragma omp parallel for
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                for (NodeID_ v : g.in_neigh(u))
                {
                    SGOffset in_offsets_local =
                        __sync_fetch_and_add(&(in_offsets[new_ids[u]]), 1);
                    in_neighs[in_offsets_local] = new_ids[v];
                }
                std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
            }
            t.Stop();
            g_relabel = CSRGraph<NodeID_, DestID_, invert>(
                            g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
        }
        else
        {
            t.Stop();
            g_relabel = CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                        out_neighs);
        }
        g_relabel.copy_org_ids(g.get_org_ids());
        g_relabel.update_org_ids(new_ids);
        PrintTime("Relabel Map Time", t.Seconds());
        return g_relabel;
    }

    /**
     * Convert ReorderingAlgo to string.
     * Delegates to the global ::ReorderingAlgoStr in reorder/reorder.h.
     */
    const std::string ReorderingAlgoStr(ReorderingAlgo type)
    {
        return ::ReorderingAlgoStr(type);
    }
    
    /**
     * Apply a basic reordering algorithm to a subgraph.
     * 
     * This is a helper for per-community reordering that handles the common
     * basic algorithms. It does NOT handle complex algorithms like Leiden,
     * GraphBrew, or AdaptiveOrder (use GenerateMappingLocalEdgelist for those).
     * 
     * @param sub_g The subgraph to reorder
     * @param sub_new_ids Output mapping (must be pre-sized to sub_g.num_nodes())
     * @param algo Algorithm to apply
     * @param useOutdeg Whether to use out-degree (true) or in-degree (false)
     */
    void ApplyBasicReordering(CSRGraph<NodeID_, DestID_, invert>& sub_g,
                              pvector<NodeID_>& sub_new_ids,
                              ReorderingAlgo algo,
                              bool useOutdeg) {
        // Delegate to standalone function in reorder_types.h
        ::ApplyBasicReorderingStandalone<NodeID_, DestID_, WeightT_, invert>(
            sub_g, sub_new_ids, algo, useOutdeg, cli_.filename());
    }
    
    /**
     * Reorder a community's nodes and assign global IDs.
     * 
     * This helper encapsulates the common pattern:
     * 1. Build global-to-local / local-to-global mappings
     * 2. Create edge list for the induced subgraph
     * 3. Build CSR graph from edge list
     * 4. Apply reordering algorithm
     * 5. Map local reordered IDs back to global IDs
     * 6. Assign sequential global IDs to the reordered nodes
     * 
     * Delegates to ::ReorderCommunitySubgraphStandalone in reorder/reorder.h
     * 
     * @param g The full graph
     * @param nodes Nodes in this community
     * @param node_set Set version for O(1) membership lookup
     * @param algo Algorithm to apply
     * @param useOutdeg Use out-degree (true) or in-degree (false)
     * @param new_ids Output global mapping (modified in place)
     * @param current_id Starting global ID (updated after assignment)
     */
    void ReorderCommunitySubgraph(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        const std::vector<NodeID_>& nodes,
        const std::unordered_set<NodeID_>& node_set,
        ReorderingAlgo algo,
        bool useOutdeg,
        pvector<NodeID_>& new_ids,
        NodeID_& current_id)
    {
        ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, nodes, node_set, algo, useOutdeg, new_ids, current_id);
    }

    void GenerateMapping(CSRGraph<NodeID_, DestID_, invert> &g,
                         pvector<NodeID_> &new_ids,
                         ReorderingAlgo reordering_algo, bool useOutdeg,
                         std::vector<std::string> reordering_options)
    {
        // Unified timing wrapper for all reordering algorithms
        Timer reorder_timer;
        reorder_timer.Start();

        // Clear staged reorder metadata before dispatch
        using namespace graphbrew::database;
        ClearStagedReorderMeta();

        switch (reordering_algo)
        {
        case HubSort:
            GenerateHubSortMapping(g, new_ids, useOutdeg);
            break;
        case Sort:
            GenerateSortMapping(g, new_ids, useOutdeg);
            break;
        case DBG:
            GenerateDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubSortDBG:
            GenerateHubSortDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubClusterDBG:
            GenerateHubClusterDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubCluster:
            GenerateHubClusterMapping(g, new_ids, useOutdeg);
            break;
        case Random:
            GenerateRandomMapping(g, new_ids);
            // RandOrder(g, new_ids, false, false);
            break;
        case RabbitOrder:
        {
            // RabbitOrder with variants: csr (default), boost
            // Format: -o 8:variant (e.g., -o 8:boost for original Boost-based)
            std::string variant = resolveVariant(reordering_options, "csr");
            
#ifdef RABBIT_ENABLE
            if (variant == "boost") {
                // Original Boost-based RabbitOrder needs preprocessing
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderMapping(g_trans, new_ids_local_2);
                
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    new_ids[n] = new_ids_local_2[new_ids_local[n]];
                }
            } else
#endif
            {
                // Native CSR implementation with degree preprocessing (like boost)
                // Pre-sort nodes by degree for better community detection convergence
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderCSRMapping(g_trans, new_ids_local_2);
                warnUnknownVariant(variant, "RabbitOrder", {"csr", "boost"});
                
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    new_ids[n] = new_ids_local_2[new_ids_local[n]];
                }
            }
        }
        break;
        case GOrder:
        {
            // GOrder with variants: default (GoGraph), csr (CSR serial), fast (parallel batch)
            // Format: -o 9:variant (e.g., -o 9:csr, -o 9:fast)
            std::string gorder_variant = resolveVariant(reordering_options);
            if (gorder_variant == "csr" || gorder_variant == "sym") {
                GenerateGOrderCSRMapping(g, new_ids);
            } else if (gorder_variant == "fast") {
                GenerateGOrderFastMapping(g, new_ids);
            } else {
                warnUnknownVariant(gorder_variant, "GOrder", {"csr", "sym", "fast"});
                GenerateGOrderMapping(g, new_ids);
            }
        }
        break;
        case COrder:
            GenerateCOrderMapping(g, new_ids);
            break;
        case RCMOrder:
        {
            // RCM with variants: default (GoGraph baseline), bnf (CSR-native BNF)
            // Format: -o 11:variant (e.g., -o 11:bnf)
            std::string rcm_variant = resolveVariant(reordering_options);
            if (rcm_variant == "bnf") {
                GenerateRCMBNFOrderMapping(g, new_ids);
            } else {
                warnUnknownVariant(rcm_variant, "RCMOrder", {"bnf"});
                GenerateRCMOrderMapping(g, new_ids);
            }
        }
        break;
        case LeidenOrder:
            // GVE-Leiden library (baseline reference) - Format: 15:resolution
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case GraphBrewOrder:
            // GraphBrew pipeline: Leiden + per-community reordering
            // Format: 12[:preset[:final_algo[:resolution[:passes[:depth[:sub]]]]]]
            // Or token mode: 12:hrab:gvecsr:0.75
            GenerateGraphBrewMappingUnified(g, new_ids, useOutdeg, reordering_options);
            break;
        case AdaptiveOrder:
            GenerateAdaptiveMapping(g, new_ids, useOutdeg, reordering_options);
            break;
        case GoGraphOrder:
            GenerateGoGraphMapping(g, new_ids, useOutdeg);
            break;
        case MAP:
            LoadMappingFromFile(g, new_ids, reordering_options);
            break;
        case ORIGINAL:
            GenerateOriginalMapping(g, new_ids);
            break;
        default:
            std::cout << "Unknown generateMapping type: " << reordering_algo
                      << std::endl;
            std::abort();
        }
        
        // Print unified reorder time for easy parsing
        reorder_timer.Stop();
        std::cout << "=== Reorder Summary ===" << std::endl;
        PrintLabel("Algorithm", ReorderingAlgoStr(reordering_algo));
        PrintTime("Reorder Time", reorder_timer.Seconds());

        // ---- Self-recording: set global hints + record reorder metadata ----
        {
            // Note: 'using namespace graphbrew::database' already in scope from above
            std::string algo_name = ReorderingAlgoStr(reordering_algo);
            double reorder_secs = reorder_timer.Seconds();
            int algo_id = static_cast<int>(reordering_algo);

            // ── MAP mode: derive real algorithm identity from .lo filename ──
            // When using pre-generated mappings (MAP, id=13), the .lo filename
            // IS the algorithm name (e.g., GORDER.lo, GraphBrewOrder_leiden.lo).
            // Also load the real reorder time from the corresponding .time file.
            if (reordering_algo == MAP && !reordering_options.empty()) {
                const std::string& lo_path = reordering_options[0];
                // Extract filename stem: /path/to/GORDER.lo → GORDER
                auto last_sep = lo_path.find_last_of("/\\");
                std::string filename = (last_sep != std::string::npos)
                    ? lo_path.substr(last_sep + 1) : lo_path;
                auto dot_pos = filename.rfind(".lo");
                if (dot_pos != std::string::npos) {
                    algo_name = filename.substr(0, dot_pos);
                }
                // Try to resolve real algorithm_id from name
                auto [found, real_algo] = lookupAlgorithm(algo_name);
                if (found) {
                    algo_id = static_cast<int>(real_algo);
                }
                // Load actual reorder time from .time file (not the .lo load time)
                if (lo_path.size() > 3) {
                    std::string time_path = lo_path.substr(0, lo_path.size() - 3) + ".time";
                    std::ifstream tf(time_path);
                    if (tf.is_open()) {
                        double file_time;
                        if (tf >> file_time && file_time > 0) {
                            reorder_secs = file_time;
                        }
                    }
                }
            }

            // Set global hints for BenchmarkKernel's RunReport
            SetReorderTimeHint(reorder_secs);
            SetReorderAlgoHint(algo_name);
            SetReorderAlgoIdHint(algo_id);

            // Build ReorderMeta hint, merging any staged algorithm-specific details
            ReorderMeta meta = GetStagedReorderMeta();
            meta.algorithm    = algo_name;
            meta.algorithm_id = algo_id;
            meta.reorder_time = reorder_secs;
            AppendReorderMetaHint(meta);
        }

        // std::cout << std::endl;
        // for (size_t i = 0; i < new_ids.size(); ++i)
        // {
        //     std::cout <<  i << "->" << new_ids[i] << " " << static_cast<size_t>(g.out_degree(i)) << std::endl;
        // }
        // std::cout << std::endl;
#ifdef _DEBUG
        VerifyMapping(g, new_ids);
        // exit(-1);
#endif
    }

    void
    GenerateMappingLocalEdgelist(const CSRGraph<NodeID_, DestID_, invert> &g_org, EdgeList &el, pvector<NodeID_> &new_ids,
                                 ReorderingAlgo reordering_algo, bool useOutdeg,
                                 std::vector<std::string> reordering_options, int numLevels = 1, bool recursion = false)
    {
        // CRITICAL: Disable nested parallelism to avoid thread explosion
        // For small subgraphs (<100K edges), nested parallelism causes massive
        // overhead (30K+ thread creates) making it 50x slower than sequential
        const size_t MIN_EDGES_FOR_PARALLEL = 100000;
        const bool is_small_subgraph = el.size() < MIN_EDGES_FOR_PARALLEL;
        
        // Save current settings
        int prev_nested = omp_get_nested();
        int prev_max_levels = omp_get_max_active_levels();
        int prev_num_threads = omp_get_max_threads();
        
        if (is_small_subgraph) {
            // For small subgraphs: run sequentially to avoid thread overhead
            omp_set_nested(0);
            omp_set_max_active_levels(1);
            omp_set_num_threads(1);
        } else {
            // For large subgraphs: enable limited parallelism
            omp_set_nested(1);
            omp_set_max_active_levels(2);  // Limit nesting depth
        }
        
        CSRGraph<NodeID_, DestID_, invert> g = MakeLocalGraphFromEL(el);
        g.copy_org_ids(g_org.get_org_ids());

        switch (reordering_algo)
        {
        case HubSort:
            GenerateHubSortMapping(g, new_ids, useOutdeg);
            break;
        case Sort:
            GenerateSortMapping(g, new_ids, useOutdeg);
            break;
        case DBG:
            GenerateDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubSortDBG:
            GenerateHubSortDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubClusterDBG:
            GenerateHubClusterDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubCluster:
            GenerateHubClusterMapping(g, new_ids, useOutdeg);
            break;
        case Random:
            GenerateRandomMapping(g, new_ids);
            // RandOrder(g, new_ids, false, false);
            break;
        case RabbitOrder:
        {
            // RabbitOrder with variants: csr (default), boost
            std::string variant = resolveVariant(reordering_options, "csr");
            
#ifdef RABBIT_ENABLE
            if (variant == "boost") {
                // Original Boost-based RabbitOrder needs preprocessing
                pvector<NodeID_> new_ids_local(g_org.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g_org.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                g = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderMapping(g, new_ids_local_2);

                // Only parallelize final merge for large graphs
                // CRITICAL: Only process nodes that were in the subgraph (have valid mapping)
                if (is_small_subgraph) {
                    for (NodeID_ n = 0; n < g_org.num_nodes(); n++)
                    {
                        if (new_ids_local[n] != (NodeID_)-1 && new_ids_local[n] < g_org.num_nodes()) {
                            new_ids[n] = new_ids_local_2[new_ids_local[n]];
                        }
                    }
                } else {
                    #pragma omp parallel for
                    for (NodeID_ n = 0; n < g_org.num_nodes(); n++)
                    {
                        if (new_ids_local[n] != (NodeID_)-1 && new_ids_local[n] < g_org.num_nodes()) {
                            new_ids[n] = new_ids_local_2[new_ids_local[n]];
                        }
                    }
                }
            } else
#endif
            {
                // Native CSR implementation with degree preprocessing (like boost)
                // Pre-sort nodes by degree for better community detection convergence
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderCSRMapping(g_trans, new_ids_local_2);
                warnUnknownVariant(variant, "RabbitOrder", {"csr", "boost"});
                
                // Combine mappings - handle subgraph case
                if (is_small_subgraph) {
                    for (NodeID_ n = 0; n < g.num_nodes(); n++)
                    {
                        new_ids[n] = new_ids_local_2[new_ids_local[n]];
                    }
                } else {
                    #pragma omp parallel for
                    for (NodeID_ n = 0; n < g.num_nodes(); n++)
                    {
                        new_ids[n] = new_ids_local_2[new_ids_local[n]];
                    }
                }
            }
        }
        break;
        case GOrder:
        {
            // GOrder with variants: default (GoGraph), csr (CSR serial), fast (parallel batch)
            // Format: -o 9:csr or -o 9:fast
            std::string gorder_variant = resolveVariant(reordering_options);
            if (gorder_variant == "csr" || gorder_variant == "sym") {
                GenerateGOrderCSRMapping(g, new_ids);
            } else if (gorder_variant == "fast") {
                GenerateGOrderFastMapping(g, new_ids);
            } else {
                warnUnknownVariant(gorder_variant, "GOrder", {"csr", "sym", "fast"});
                GenerateGOrderMapping(g, new_ids);
            }
        }
        break;
        case COrder:
            GenerateCOrderMapping(g, new_ids);
            break;
        case RCMOrder:
        {
            // RCM with variants: default (GoGraph baseline), bnf (CSR-native BNF)
            std::string rcm_variant = resolveVariant(reordering_options);
            if (rcm_variant == "bnf") {
                GenerateRCMBNFOrderMapping(g, new_ids);
            } else {
                warnUnknownVariant(rcm_variant, "RCMOrder", {"bnf"});
                GenerateRCMOrderMapping(g, new_ids);
            }
        }
        break;
        case LeidenOrder:
            // GVE-Leiden library (baseline reference) - Format: 15:resolution
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case GraphBrewOrder:
            GenerateGraphBrewMappingUnified(g, new_ids, useOutdeg, reordering_options);
            break;
        case AdaptiveOrder:
            GenerateAdaptiveMapping(g, new_ids, useOutdeg, reordering_options);
            break;
        case GoGraphOrder:
            GenerateGoGraphMapping(g, new_ids, useOutdeg);
            break;
        case MAP:
            LoadMappingFromFile(g, new_ids, reordering_options);
            break;
        case ORIGINAL:
            GenerateOriginalMapping(g, new_ids);
            break;
        default:
            std::cout << "Unknown generateMapping type: " << reordering_algo
                      << std::endl;
            std::abort();
        }
        
        // Restore OpenMP settings
        omp_set_nested(prev_nested);
        omp_set_max_active_levels(prev_max_levels);
        omp_set_num_threads(prev_num_threads);
        
#ifdef _DEBUG
        VerifyMapping(g, new_ids);
        // exit(-1);
#endif
    }

    /**
     * Convert string argument to ReorderingAlgo.
     * Delegates to the global ::getReorderingAlgo in reorder/reorder.h.
     */
    ReorderingAlgo getReorderingAlgo(const char *arg)
    {
        return ::getReorderingAlgo(arg);
    }

    void VerifyMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                       const pvector<NodeID_> &new_ids)
    {
        NodeID_ *hist = alloc_align_4k<NodeID_>(g.num_nodes());
        int64_t num_nodes = g.num_nodes();

        #pragma omp parallel for
        for (long i = 0; i < num_nodes; i++)
        {
            hist[i] = new_ids[i];
        }

        __gnu_parallel::stable_sort(&hist[0], &hist[num_nodes]);

        NodeID_ count = 0;

        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            if (hist[i] != i)
            {
                __sync_fetch_and_add(&count, 1);
            }
        }

        if (count != 0)
        {
            std::cout << "Num of vertices did not match: " << count << std::endl;
            std::cout << "Mapping is invalid.!" << std::endl;
            std::abort();
        }
        else
        {
            std::cout << "Mapping is valid.!" << std::endl;
        }
        std::free(hist);
    }

    void printReorderingMethods(const std::string &filename, Timer t)
    {
        std::size_t last_slash = filename.rfind('/');
        std::string basename = filename.substr(last_slash + 1);
        std::size_t last_dot = basename.rfind('.');
        std::string stem = basename.substr(0, last_dot);

        std::vector<int> codes;
        std::istringstream iss(stem);
        std::string part;
        while (getline(iss, part, '_'))
        {
            int num;
            if (std::istringstream(part) >> num)
            {
                codes.push_back(num);
            }
        }

        // std::cout << "Reordering methods for file '" << filename
        //           << "':" << std::endl;
        for (int code : codes)
        {
            try
            {
                std::string algoStr =
                    ReorderingAlgoStr(static_cast<ReorderingAlgo>(code)) + " Map Time";
                PrintTime(algoStr, t.Seconds());
            }
            catch (...)
            {
                std::cerr << "Invalid code: " << code << std::endl;
            }
        }
    }

    void LoadMappingFromFile(const CSRGraph<NodeID_, DestID_, invert> &g,
                             pvector<NodeID_> &new_ids,
                             std::vector<std::string> reordering_options)
    {
        Timer t;
        int64_t num_nodes = g.num_nodes();
        std::string map_file = "mapping.lo";

        // std::cout << "Options: ";
        // for (const auto& param : reordering_options) {
        //   std::cout << param << " ";
        // }
        // std::cout << std::endl;

        if (!reordering_options.empty())
            map_file = reordering_options[0];

        t.Start();
        std::ifstream ifs(map_file, std::ifstream::in);
        if (!ifs.is_open())
        {
            std::cerr << "File " << map_file << " does not exist!" << std::endl;
            throw std::runtime_error("File not found.");
        }
        std::string file_suffix = map_file.substr(map_file.find_last_of('.'));
        if (file_suffix != ".so" && file_suffix != ".lo")
        {
            std::cerr << "Unsupported file format: " << file_suffix << std::endl;
            throw std::invalid_argument("Unsupported format.");
        }
        // The .lo/.so file stores the INVERSE mapping written by
        // WriteListLabels from org_ids_:
        //     file[new_id] = original_id
        // where original_id traces all the way back to the source graph
        // (i.e. the MTX / edge-list vertex numbering), NOT necessarily
        // the internal vertex IDs of the .sg file we just loaded.
        //
        // RelabelByMapping() expects the FORWARD mapping indexed by the
        // loaded graph's internal IDs:
        //     new_ids[sg_id] = new_id
        //
        // If the .sg was previously reordered (sg has non-identity org_ids),
        // there are two coordinate spaces:
        //   sg_id  — internal vertex index in the loaded .sg graph
        //   org_id — original vertex ID (from the source MTX/el)
        //
        // The graph stores: g.org_ids[sg_id] = org_id
        //
        // Steps:
        //   1. Read file → inverse_ids[new_id] = org_id
        //   2. Invert   → org_to_new[org_id]  = new_id
        //   3. Compose with graph's org_ids:
        //        new_ids[sg_id] = org_to_new[ g.org_ids[sg_id] ]
        //      This converts from org_id-space into sg_id-space.
        //      When org_ids is identity (no prior reorder), step 3 is a no-op.
        pvector<NodeID_> inverse_ids(num_nodes);
        NodeID_ *label_ids = new NodeID_[num_nodes];
        if (file_suffix == ".so")
        {
            ifs.read(reinterpret_cast<char *>(label_ids),
                     g.num_nodes() * sizeof(NodeID_));
            #pragma omp parallel for
            for (int64_t i = 0; i < num_nodes; i++)
            {
                inverse_ids[i] = label_ids[i];
            }
        }
        else
        {
            for (int64_t i = 0; i < num_nodes; i++)
            {
                ifs >> inverse_ids[i];
            }
        }
        delete[] label_ids;
        ifs.close();

        // Step 2: Invert  inverse_ids[new_id] = org_id  →  org_to_new[org_id] = new_id
        pvector<NodeID_> org_to_new(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            org_to_new[inverse_ids[i]] = i;
        }

        // Step 3: Compose with graph's org_ids to map sg_id → new_id
        NodeID_ *org_ids = g.get_org_ids();
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            new_ids[i] = org_to_new[org_ids[i]];
        }
        t.Stop();

        printReorderingMethods(map_file, t);
        PrintTime("Load Map Time", t.Seconds());
    }

    /**
     * @brief Identity mapping - keeps original vertex IDs
     * Delegates to ::GenerateOriginalMapping in reorder/reorder_basic.h
     */
    void GenerateOriginalMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids) {
        ::GenerateOriginalMapping<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * @brief Random permutation of vertex IDs
     * Delegates to ::GenerateRandomMapping in reorder/reorder_basic.h
     */
    void GenerateRandomMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids) {
        ::GenerateRandomMapping<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * @brief Random permutation using atomic compare-and-swap (legacy v2)
     * Delegates to ::GenerateRandomMapping_v2 in reorder/reorder_basic.h
     */
    void GenerateRandomMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids) {
        ::GenerateRandomMapping_v2<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * @brief HubSort within DBG buckets
     * Delegates to ::GenerateHubSortDBGMapping in reorder/reorder_hub.h
     */
    void GenerateHubSortDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubSortDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief HubCluster within DBG buckets (2-bucket version)
     * Delegates to ::GenerateHubClusterDBGMapping in reorder/reorder_hub.h
     */
    void GenerateHubClusterDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                      pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubClusterDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Sort vertices by degree, placing hubs first
     * Delegates to ::GenerateHubSortMapping in reorder/reorder_hub.h
     */
    void GenerateHubSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Sort all vertices by degree
     * Delegates to ::GenerateSortMapping in reorder/reorder_basic.h
     */
    void GenerateSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                             pvector<NodeID_> &new_ids, bool useOutdeg,
                             bool lesser = false) {
        ::GenerateSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg, lesser);
    }

    /**
     * Sort by (out-degree, in-degree) descending with RabbitOrder-style handling
     * Delegates to ::GenerateSortMappingRabbit in reorder/reorder_basic.h
     */
    void GenerateSortMappingRabbit(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg,
                                   bool lesser = false) {
        ::GenerateSortMappingRabbit<NodeID_, DestID_, invert>(g, new_ids, useOutdeg, lesser);
    }

    /**
     * @brief Degree-Based Grouping into logarithmic buckets
     * Delegates to ::GenerateDBGMapping in reorder/reorder_hub.h
     */
    void GenerateDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                            pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Cluster hub vertices with partition-aware locality
     * Delegates to ::GenerateHubClusterMapping in reorder/reorder_hub.h
     */
    void GenerateHubClusterMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubClusterMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Cache-aware workload balancing ordering
     * Delegates to ::GenerateCOrderMapping in reorder/reorder_classic.h
     */
    void GenerateCOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids) {
        ::GenerateCOrderMapping<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * COrder v2 - Optimized parallel version using Vector2d
     * Delegates to ::GenerateCOrderMapping_v2 in reorder/reorder_classic.h
     */
    void GenerateCOrderMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids) {
        ::GenerateCOrderMapping_v2<NodeID_, DestID_, invert>(g, new_ids);
    }


#ifdef RABBIT_ENABLE
    /**
     * @brief RabbitOrder - Community-aware reordering using hierarchical clustering
     * Delegates to ::GenerateRabbitOrderMapping in reorder/reorder_rabbit.h
     */
    void GenerateRabbitOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                    pvector<NodeID_> &new_ids) {
        ::GenerateRabbitOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids);
    }

    /**
     * @brief Compute RabbitOrder modularity from edge list
     * Delegates to ::GenerateRabbitModularityEdgelist in reorder/reorder_rabbit.h
     */
    double GenerateRabbitModularityEdgelist(EdgeList &edgesList, bool is_weighted) {
        std::vector<edge_list::edge> edges(edgesList.size());
        #pragma omp parallel for
        for (size_t i = 0; i < edges.size(); ++i) {
            if (is_weighted) {
                edges[i] = std::make_tuple(
                    static_cast<rabbit_order::vint>(edgesList[i].u),
                    static_cast<NodeWeight<NodeID_, WeightT_>>(edgesList[i].v).v,
                    static_cast<NodeWeight<NodeID_, WeightT_>>(edgesList[i].v).w);
            } else {
                edges[i] = std::make_tuple(
                    static_cast<rabbit_order::vint>(edgesList[i].u),
                    static_cast<rabbit_order::vint>(edgesList[i].v),
                    1.0f);
            }
        }
        return ::GenerateRabbitModularityEdgelist<NodeID_>(edges);
    }
    /**
     * @brief RabbitOrder from edge list
     * Delegates to ::GenerateRabbitOrderMappingEdgelist in reorder/reorder_rabbit.h
     */
    void GenerateRabbitOrderMappingEdgelist(const std::vector<edge_list::edge> &edges,
                                            pvector<NodeID_> &new_ids) {
        ::GenerateRabbitOrderMappingEdgelist<NodeID_>(edges, new_ids);
    }
#endif

    /*
       MIT License

       Copyright (c) 2016, Hao Wei.

       Permission is hereby granted, free of charge, to any person obtaining a
       copy of this software and associated documentation files (the "Software"),
       to deal in the Software without restriction, including without limitation
       the rights to use, copy, modify, merge, publish, distribute, sublicense,
       and/or sell copies of the Software, and to permit persons to whom the
       Software is furnished to do so, subject to the following conditions:

       The above copyright notice and this permission notice shall be included in
       all copies or substantial portions of the Software.

       THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
       IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
       FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
       AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
       LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
       FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
       DEALINGS IN THE SOFTWARE.
     */

    /**
     * @brief GOrder - Graph Ordering using dynamic programming and windowing
     * Delegates to ::GenerateGOrderMapping in reorder/reorder_classic.h
     */
    void GenerateGOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids) {
        ::GenerateGOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
    }

    /**
     * @brief GOrder CSR variant — CSR-native GOrder
     * Delegates to ::GenerateGOrderCSRMapping in reorder/reorder_gorder.h
     * Accessed via: -o 9:csr (sym is an alias for csr)
     */
    void GenerateGOrderCSRMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids) {
        ::GenerateGOrderCSRMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
    }

    /**
     * @brief GOrder fast variant — parallel batch GOrder
     * Delegates to ::GenerateGOrderFastMapping in reorder/reorder_gorder.h
     * Accessed via: -o 9:fast
     */
    void GenerateGOrderFastMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids) {
        ::GenerateGOrderFastMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
    }

    /**
     * @brief Reverse Cuthill-McKee ordering for bandwidth reduction (baseline)
     * Delegates to ::GenerateRCMOrderMapping in reorder/reorder_classic.h
     */
    void GenerateRCMOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids) {
        ::GenerateRCMOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
    }

    /**
     * @brief RCM BNF variant — CSR-native RCM with BNF start + level-parallel BFS
     * Delegates to ::GenerateRCMBNFOrderMapping in reorder/reorder_rcm.h
     * Accessed via: -o 11:bnf
     */
    void GenerateRCMBNFOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                    pvector<NodeID_> &new_ids) {
        ::GenerateRCMBNFOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
    }

    // HELPERS
    // -------

    template <class G, class R>
    inline double getModularity(const G &x, const R &a, double M)
    {
        auto fc = [&](auto u)
        {
            return a.membership[u];
        };
        return modularityByOmp(x, fc, M, 1.0);
    }

    template <class K, class W>
    inline float refinementTime(const LouvainResult<K, W> &a)
    {
        return 0;
    }
    template <class K, class W>
    inline float refinementTime(const LeidenResult<K, W> &a)
    {
        return a.refinementTime;
    }

    // PERFORM EXPERIMENT
    // ------------------

    template <class G>
    void runExperiment(G &x, double resolution = 0.75, int maxIterations = 10,
                       int maxPasses = 10)
    {
        // using K = typename G::key_type;
        // using V = typename G::edge_value_type;
        Timer tm;
        std::random_device dev;
        std::default_random_engine rnd(dev());
        int repeat = REPEAT_METHOD;
        double M = edgeWeightOmp(x) / 2;
        // Follow a specific result logging format, which can be easily parsed
        // later.
        // auto flog = [&](const auto &ans, const char *technique) {
        //   printf("{%09.1fms, %09.1fms mark, %09.1fms init, %09.1fms "
        //          "firstpass,%09.1fms locmove, %09.1fms refine, %09.1fms aggr,
        //          %.3e " "aff,%04d iters, %03d passes, %01.9f modularity, "
        //          "%zu/%zudisconnected} %s\n",
        //          ans.time, ans.markingTime, ans.initializationTime,
        //          ans.firstPassTime, ans.localMoveTime, refinementTime(ans),
        //          ans.aggregationTime, double(ans.affectedVertices),
        //          ans.iterations, ans.passes, getModularity(x, ans, M),
        //          countValue(communitiesDisconnectedOmp(x, ans.membership),
        //          char(1)), communities(x, ans.membership).size(), technique);
        // };
        // Get community memberships on original graph (static).
        {
            // auto a0 = louvainStaticOmp(x, {repeat});
            // flog(a0, "louvainStaticOmp");
        }
        {

            tm.Start();

            // auto b0 = leidenStaticOmp<false, false>(rnd, x, {repeat});
            // flog(b0, "leidenStaticOmpGreedy");
            // auto b1 = leidenStaticOmp<false,  true>(rnd, x, {repeat});
            // flog(b1, "leidenStaticOmpGreedyOrg");
            // auto c0 = leidenStaticOmp<false, false>(
            //   rnd, x, {repeat, 0.5, 1e-12, 0.8, 1.0, 100, 100});
            // flog(c0, "leidenStaticOmpGreedyMedium");
            auto c1 = leidenStaticOmp<false, false>(
                          rnd, x,
            {repeat, resolution, 1e-12, 0.8, 1.0, maxIterations, maxPasses});
            tm.Stop();
            PrintTime("Modularity", getModularity(x, c1, M));
            PrintTime("LeidenOrder Map Time", tm.Seconds());

            // flog(c1, "leidenStaticOmpGreedyMediumOrg");
            // auto d0 = leidenStaticOmp<false, false>(rnd, x, {repeat, 1.0,
            // 1e-12, 1.0, 1.0, 100, 100}); flog(d0, "leidenStaticOmpGreedyHeavy");
            // auto d1 = leidenStaticOmp<false,  true>(rnd, x, {repeat, 1.0,
            // 1e-12, 1.0, 1.0, 100, 100}); flog(d1, "leidenStaticOmpGreedyHeavyOrg");
        }
        {
            // auto b2 = leidenStaticOmp<true, false>(rnd, x, {repeat});
            // flog(b2, "leidenStaticOmpRandom");
            // auto b3 = leidenStaticOmp<true,  true>(rnd, x, {repeat});
            // flog(b3, "leidenStaticOmpRandomOrg");
            // auto c2 = leidenStaticOmp<true, false>(rnd, x, {repeat, 1.0, 1e-12,
            // 0.8, 1.0, 100, 100}); flog(c2, "leidenStaticOmpRandomMedium"); auto c3
            // = leidenStaticOmp<true,  true>(rnd, x, {repeat, 1.0, 1e-12, 0.8, 1.0,
            // 100, 100}); flog(c3, "leidenStaticOmpRandomMediumOrg"); auto d2 =
            // leidenStaticOmp<true, false>(rnd, x, {repeat, 1.0, 1e-12, 1.0, 1.0,
            // 100, 100}); flog(d2, "leidenStaticOmpRandomHeavy"); auto d3 =
            // leidenStaticOmp<true,  true>(rnd, x, {repeat, 1.0, 1e-12, 1.0, 1.0,
            // 100, 100}); flog(d3, "leidenStaticOmpRandomHeavyOrg");
        }
    }

    using K = uint32_t;

    //==========================================================================
    // GVE-LEIDEN: True Leiden Algorithm on CSR Graphs
    // Implementation following ACM paper: "Fast Leiden Algorithm for Community
    // Detection in Shared Memory Setting" (DOI: 10.1145/3673038.3673146)
    //
    // IMPORTANT: This implementation handles BOTH symmetric and non-symmetric
    // CSR graphs by scanning both out_neigh and in_neigh to get all edges.
    //
    // Key differences from Louvain:
    // 1. Refinement phase: Only isolated vertices can move
    // 2. Community bounds: Refined communities constrained within local-moving results
    // 3. Well-connected communities guaranteed
    //
    // Note: GVELeidenResult, GVEDendroResult, GVEAtomicDendroResult structures,
    // atomicMergeToDendro, mergeToDendro, and initDendrogram are now in reorder/reorder_types.h
    //==========================================================================
    
    /**
     * Traverse dendrogram using DFS, assigning new IDs.
     * Orders vertices so that community members are contiguous.
     * Hub-first: Higher weight children are visited first.
     * Delegates to ::traverseDendrogramDFS in reorder/reorder_types.h
     */
    template <typename K = uint32_t>
    void traverseDendrogramDFS(
        const GVEDendroResult<K>& dendro,
        pvector<NodeID_>& new_ids,
        bool hub_first = true) {
        
        // Convert pvector to std::vector for the standalone function
        std::vector<NodeID_> temp_ids(new_ids.size());
        ::traverseDendrogramDFS<K, NodeID_>(dendro, temp_ids, hub_first);
        
        // Copy back to pvector
        #pragma omp parallel for
        for (size_t i = 0; i < temp_ids.size(); ++i) {
            new_ids[i] = temp_ids[i];
        }
    }
    
    /**
     * Fast parallel modularity computation for any community assignment.
     * 
     * Modularity Q = (1/2m) * Σ[A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
     * 
     * Delegates to ::computeModularityCSR in reorder/reorder_types.h
     */
    template <typename K>
    double computeModularityCSR(
        const CSRGraph<NodeID_, DestID_, true>& g,
        const std::vector<K>& community,
        double resolution = 1.0) {
        return ::computeModularityCSR<K, NodeID_, DestID_>(g, community, resolution);
    }
    
    /**
     * Scan all edges connected to vertex u (both out-edges and in-edges).
     * Delegates to ::scanVertexEdges in reorder/reorder_types.h
     */
    template <typename K, typename W>
    inline W scanVertexEdges(
        NodeID_ u,
        const K* vcom,
        std::unordered_map<K, W>& hash,
        K d,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        return ::scanVertexEdges<K, W, NodeID_, DestID_>(u, vcom, hash, d, g, graph_is_symmetric);
    }
    
    /**
     * Compute vertex total weight (degree sum for unweighted graphs).
     * Delegates to ::computeVertexTotalWeightCSR in reorder/reorder_types.h
     */
    template <typename W>
    inline W computeVertexTotalWeight(
        NodeID_ u,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        return ::computeVertexTotalWeightCSR<W, NodeID_, DestID_>(u, g, graph_is_symmetric);
    }
    
    /**
     * Mark all neighbors of vertex u as affected.
     * Delegates to ::markNeighborsAffected in reorder/reorder_types.h
     */
    inline void markNeighborsAffected(
        NodeID_ u,
        std::vector<char>& vaff,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        ::markNeighborsAffected<NodeID_, DestID_>(u, vaff, g, graph_is_symmetric);
    }
    
    //==========================================================================
    // LEIDENFAST: Parallel Community-Based Graph Reordering
    //==========================================================================
    
    /**
     * LeidenFast: Fast parallel reordering using Union-Find + Label Propagation
     * 
     * Improvements over initial version:
     * 1. Parallel Union-Find with atomic CAS (like RabbitOrder)
     * 2. Best-fit merging (scan ALL neighbors, not first-fit)
     * 3. Efficient hash-based counting for label propagation
     * 4. Multi-level aggregation for deeper hierarchy
     * 5. Modularity-weighted edge selection
     * 
     * Strategy:
     * Phase 1: Parallel Union-Find merging with modularity criterion
     * Phase 2: Label propagation refinement with modularity-weighted moves
     * Phase 3: Optional aggregation for hierarchical structure
     * Final: Order by community strength DESC, degree DESC within community
     */
    
    //==========================================================================
    // LEIDEN AUTO-RESOLUTION
    //==========================================================================
    
    /**
     * Compute optimal resolution based on graph properties.
     * 
     * Heuristic for stable partitions for reordering; not a research-derived
     * optimum. Users should sweep γ for best community quality.
     */
    template<typename NodeID_T, typename DestID_T>
    double LeidenAutoResolution(const CSRGraph<NodeID_T, DestID_T, true>& g) {
        return computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    
    
    void sort_by_vector_element(
        std::vector<std::vector<K>> &communityVectorTuplePerPass,
        size_t element_index)
    {
        __gnu_parallel::stable_sort(
            communityVectorTuplePerPass.begin(), communityVectorTuplePerPass.end(),
            [&](const std::vector<K> &a, const std::vector<K> &b)
        {
            return a[element_index] < b[element_index];
        });
    }

    void GenerateLeidenMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids,
                               std::vector<std::string> reordering_options)
    {

        Timer tm;

        using V = TYPE;
        install_sigsegv();

        // Use auto-resolution based on graph density
        double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        // Unified defaults across all Leiden algorithms for fair comparison
        int maxIterations = LEIDEN_DEFAULT_ITERATIONS;
        int maxPasses = LEIDEN_DEFAULT_PASSES;

        if (!reordering_options.empty() && !reordering_options[0].empty())
        {
            const std::string& res_opt = reordering_options[0];
            // Handle special keywords (auto, dynamic, etc.)
            if (res_opt == "auto" || res_opt == "0" || res_opt.rfind("dynamic", 0) == 0) {
                // Keep auto-resolution
            } else {
                try {
                    double parsed = std::stod(res_opt);
                    if (parsed > 0 && parsed <= 3) {
                        resolution = parsed;
                    }
                } catch (...) {
                    // Parse error, keep auto-resolution
                }
            }
        }
        if (reordering_options.size() > 1 && !reordering_options[1].empty())
        {
            try { maxIterations = std::stoi(reordering_options[1]); } catch (...) {}
        }
        if (reordering_options.size() > 2 && !reordering_options[2].empty())
        {
            try { maxPasses = std::stoi(reordering_options[2]); } catch (...) {}
        }

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();

        std::vector<std::tuple<size_t, size_t, double>> edges(num_edges);
        edges.reserve(num_edges);
        // Parallel loop to construct the edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            NodeID_ out_start = g.out_offset(i);

            NodeID_ j = 0;
            for (DestID_ neighbor : g.out_neigh(i))
            {
                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    WeightT_ weight =
                        static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;

                    std::tuple<size_t, size_t, double> edge =
                        std::make_tuple(i, dest, weight);
                    edges[out_start + j] = edge;
                }
                else
                {
                    std::tuple<size_t, size_t, double> edge =
                        std::make_tuple(i, neighbor, 1.0f);
                    edges[out_start + j] = edge;
                }
                ++j;
            }
        }

        tm.Start();
        bool symmetric = false;
        bool weighted = g.is_weighted();
        DiGraph<K, None, V> x;
        readVecOmpW(x, edges, num_nodes, symmetric,
                    weighted); // LOG(""); println(x);
        edges.clear();
        x = symmetricizeOmp(x);

        tm.Stop();
        PrintTime("DiGraph graph", tm.Seconds());

        runExperiment(x, resolution, maxIterations, maxPasses);

        size_t num_nodesx;
        size_t num_passes;
        num_nodesx = x.span();
        num_passes = x.communityMappingPerPass.size() + 2;

        // Use flat array with stride for better cache locality (SoA pattern)
        // Layout: [all node IDs][all degrees][pass0 communities][pass1 communities]...
        const size_t stride = num_nodesx;
        std::vector<K> communityDataFlat(num_nodesx * num_passes);
        
        // Initialize node IDs and degrees
        tm.Start();
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            communityDataFlat[i] = i;                        // column 0: node ID
            communityDataFlat[stride + i] = x.degree(i);     // column 1: degree
        }

        // Copy community mappings per pass
        for (size_t p = 0; p < num_passes - 2; ++p)
        {
            K* dest_col = &communityDataFlat[(2 + p) * stride];
            const auto& src = x.communityMappingPerPass[p];
            #pragma omp parallel for
            for (size_t j = 0; j < num_nodesx; ++j)
            {
                dest_col[j] = src[j];
            }
        }

        // Sort by last pass community - create index array for indirect sort
        std::vector<size_t> sort_indices(num_nodesx);
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i) {
            sort_indices[i] = i;
        }
        
        /**
         * DENDROGRAM-BASED ORDERING (Optimization inspired by RabbitOrder)
         * 
         * Key insight: RabbitOrder outperforms LeidenOrder because it preserves
         * hierarchical locality through dendrogram DFS traversal.
         * 
         * Original LeidenOrder problem:
         *   - Only sorts by LAST pass community
         *   - Within a community, order is arbitrary
         *   - Loses fine-grained locality from earlier passes
         * 
         * Solution: Multi-level hierarchical sort
         *   - Sort by ALL passes in order: (pass_N, pass_N-1, ..., pass_0, degree)
         *   - This is equivalent to DFS traversal of the community dendrogram
         *   - Vertices in the same sub-sub-community become adjacent
         *   - Secondary sort by degree puts hubs together (cache-friendly)
         * 
         * This achieves RabbitOrder-like locality while using Leiden's
         * higher-quality community structure.
         */
        const size_t actual_passes = num_passes - 2;  // Exclude nodeID and degree columns
        
        // Sort by ALL passes (coarsest to finest) then by degree
        // This achieves dendrogram DFS-like ordering without building the tree
        __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
            [&communityDataFlat, stride, actual_passes](size_t a, size_t b) {
                // Compare all passes from coarsest (last) to finest (first)
                for (size_t p = actual_passes; p > 0; --p) {
                    size_t pass_col = 2 + p - 1;  // Column index for this pass
                    K comm_a = communityDataFlat[pass_col * stride + a];
                    K comm_b = communityDataFlat[pass_col * stride + b];
                    if (comm_a != comm_b) {
                        return comm_a < comm_b;
                    }
                }
                // All passes equal - sort by degree (descending) for hub locality
                K deg_a = communityDataFlat[stride + a];  // degree column
                K deg_b = communityDataFlat[stride + b];
                return deg_a > deg_b;  // High degree first (hubs together)
            });

        // Assign new IDs based on sorted order
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            new_ids[communityDataFlat[sort_indices[i]]] = (NodeID_)i;
        }

        // Count unique communities in last pass
        size_t num_communities = 0;
        if (!x.communityMappingPerPass.empty()) {
            const auto& last_pass = x.communityMappingPerPass.back();
            std::set<K> unique_comms(last_pass.begin(), last_pass.end());
            num_communities = unique_comms.size();
        }

        tm.Stop();
        PrintTime("GenID Time", tm.Seconds());
        PrintTime("Num Passes", x.communityMappingPerPass.size());
        PrintTime("Num Communities", num_communities);
        PrintTime("Resolution", resolution);

        // Stage reorder metadata for GenerateMapping's ReorderMeta hint
        {
            using namespace graphbrew::database;
            auto& staged = GetStagedReorderMeta();
            staged.num_passes      = static_cast<int>(x.communityMappingPerPass.size());
            staged.num_communities = static_cast<int>(num_communities);
            staged.resolution      = resolution;
        }
    }

    // ========================================================================
    // GVE-Rabbit Hybrid Algorithm
    // 
    // Combines RabbitOrder's fast incremental aggregation with Leiden's 
    // refinement phase for high-quality communities at near-RabbitOrder speed.
    //
    // Algorithm:
    // 1. FAST AGGREGATION (RabbitOrder-style):
    //    - Process vertices in degree order
    //    - For each vertex, find best neighbor maximizing ΔQ
    //    - Merge via lock-free union-find with CAS
    //    - Build dendrogram during merging
    //
    // 2. LEIDEN REFINEMENT (GVE-style):
    //    - For each community from step 1, run local refinement
    //    - Only isolated vertices (within their community bound) can move
    //    - This breaks poorly-connected clusters
    //
    // 3. ORDERING (RabbitOrder-style):
    //    - DFS traversal of dendrogram
    //    - Hub-first within each community
    // ========================================================================

    /**
     * GVE-Rabbit Result Structure
     * NOTE: GVERabbitResult is in reorder/reorder_types.h
     */
    template <typename K = uint32_t>
    using GVERabbitResult = ::GVERabbitResult<K>;

    // NOTE: GenerateGraphBrewMapping has been removed.
    // GraphBrew functionality is now in GenerateGraphBrewMappingUnified() (algo 12).

    // ========================================================================
    // RabbitOrderCSR - Native CSR implementation of Rabbit Order
    // 
    // Types and helper functions (RabbitCSRAtomPacked, RabbitCSRVertex, 
    // RabbitCSRGraph, rabbitCSRTraceCom, rabbitCSRCompactEdges, rabbitCSRUnite,
    // rabbitCSRFindBest, rabbitCSRMerge, rabbitCSRAggregate, rabbitCSRDescendants,
    // rabbitCSRComputePerm, rabbitCSRComputeModularityCSR) are now in:
    //   reorder/reorder_rabbit.h
    // ========================================================================


    /**
     * @brief RabbitOrderCSR - Native CSR implementation of Rabbit Order
     * Delegates to ::GenerateRabbitOrderCSRMapping in reorder/reorder_rabbit.h
     */
    void GenerateRabbitOrderCSRMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                       pvector<NodeID_>& new_ids) {
        ::GenerateRabbitOrderCSRMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids);
    }


    // CommunityFeatures is now defined in reorder/reorder_types.h
    // Alias for backward compatibility within BuilderBase
    using CommunityFeatures = ::CommunityFeatures;

    /**
     * Compute and print global graph topology features.
     * 
     * This computes features ONCE for the entire graph during the build phase,
     * BEFORE any reordering algorithm runs. These features describe the graph
     * structure itself and are used for:
     * 1. Perceptron-based algorithm selection
     * 2. Graph type detection
     * 3. Training weight data for Python scripts
     * 
     * Features computed:
     * - clustering_coeff: Global clustering coefficient (sampled)
     * - avg_path_length: Estimated average shortest path length
     * - diameter: Estimated graph diameter
     * - community_count: Number of communities (via fast Leiden)
     * - degree_variance: Normalized coefficient of variation of degrees
     * - hub_concentration: Fraction of edges from top 10% degree nodes
     * 
     * Output format (for Python parsing):
     *   Graph Topology Features:
     *   Clustering Coefficient: 0.1234
     *   Avg Path Length:        5.6789
     *   Diameter Estimate:      12
     *   Community Count:        45
     */
    void ComputeAndPrintGlobalTopologyFeatures(const CSRGraph<NodeID_, DestID_, invert>& g) {
        Timer t;
        t.Start();
        
        const int64_t num_nodes = g.num_nodes();
        const int64_t num_edges = g.num_edges_directed();
        
        if (num_nodes < 100) {
            // Too small for meaningful topology analysis — but still
            // record basic graph dimensions when self-recording is on.
            if (graphbrew::database::SelfRecordingEnabled()) {
                graphbrew::database::GraphProperties props;
                props.graph_name = GetGraphNameHint();
                props.nodes      = num_nodes;
                props.edges      = num_edges;
                props.avg_degree = (num_nodes > 0) ? static_cast<double>(num_edges) / num_nodes : 0.0;
                props.density    = (num_nodes > 1)
                    ? static_cast<double>(num_edges) / (static_cast<double>(num_nodes) * (num_nodes - 1))
                    : 0.0;
                graphbrew::database::BenchmarkDatabase::Get().update_graph_props(props);
            }
            return;
        }
        
        // ============================================================
        // 1. DEGREE STATISTICS (use shared utility function)
        // ============================================================
        auto deg_features = ::ComputeSampledDegreeFeatures(g, 0, true);
        double avg_degree = deg_features.avg_degree;
        double degree_variance = deg_features.degree_variance;
        double hub_concentration = deg_features.hub_concentration;
        double clustering_coeff = deg_features.clustering_coeff;
        double packing_factor = deg_features.packing_factor;
        double packing_factor_cl = deg_features.packing_factor_cl;
        double forward_edge_fraction = deg_features.forward_edge_fraction;
        double working_set_ratio = deg_features.working_set_ratio;
        double vertex_significance_skewness = deg_features.vertex_significance_skewness;
        double window_neighbor_overlap = deg_features.window_neighbor_overlap;
        double sampled_locality_score = deg_features.sampled_locality_score;
        
        // ============================================================
        // 2. DIAMETER & AVG PATH LENGTH (single BFS from high-degree node)
        // ============================================================
        double avg_path_length = 0.0;
        int diameter_estimate = 0;
        
        if (num_nodes >= 500 && num_nodes <= 10000000) {  // Skip for very large graphs
            // Find highest degree node as BFS starting point
            const size_t SAMPLE_SIZE = std::min(static_cast<size_t>(5000), static_cast<size_t>(num_nodes));
            NodeID_ start_node = 0;
            int64_t max_deg = 0;
            for (size_t i = 0; i < std::min(SAMPLE_SIZE, static_cast<size_t>(num_nodes)); ++i) {
                NodeID_ node = (static_cast<size_t>(num_nodes) > SAMPLE_SIZE) ? 
                    static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
                if (g.out_degree(node) > max_deg) {
                    max_deg = g.out_degree(node);
                    start_node = node;
                }
            }
            
            // BFS with early termination for large graphs
            std::vector<int> dist(num_nodes, -1);
            std::queue<NodeID_> bfs_queue;
            bfs_queue.push(start_node);
            dist[start_node] = 0;
            
            double path_sum = 0.0;
            size_t path_count = 0;
            int max_dist = 0;
            const size_t MAX_BFS_VISITS = std::min(static_cast<size_t>(100000), static_cast<size_t>(num_nodes));
            size_t visits = 0;
            
            while (!bfs_queue.empty() && visits < MAX_BFS_VISITS) {
                NodeID_ curr = bfs_queue.front();
                bfs_queue.pop();
                
                for (DestID_ neighbor : g.out_neigh(curr)) {
                    NodeID_ dest = static_cast<NodeID_>(neighbor);
                    if (dist[dest] == -1) {
                        dist[dest] = dist[curr] + 1;
                        bfs_queue.push(dest);
                        path_sum += dist[dest];
                        ++path_count;
                        ++visits;
                        if (dist[dest] > max_dist) {
                            max_dist = dist[dest];
                        }
                    }
                }
            }
            
            avg_path_length = (path_count > 0) ? path_sum / path_count : 1.0;
            diameter_estimate = max_dist;
        } else if (num_nodes > 10000000) {
            // For very large graphs, use rough estimates
            avg_path_length = std::log2(num_nodes);  // Small-world approximation
            diameter_estimate = static_cast<int>(avg_path_length * 2);
        }
        
        // ============================================================
        // 3. FAST MODULARITY (Afforest-style subgraph-sampled Louvain)
        // ============================================================
        // Hub-capped Louvain: O(ROUNDS × N × HUB_CAP) local-moving
        // + O(N × HUB_CAP) sampled Q — sublinear in m for power-law graphs.
        //
        // This is OPTIONAL because:
        //  - C++ runtime perceptron (scoreBase) uses estimated_modularity =
        //    min(0.9, CC * 1.5), NOT this value.
        //  - Training (Phase 4) also uses the CC*1.5 estimate for consistency.
        //  - On twitter7 / com-Friendster this takes 25-44s — significant
        //    overhead for what amounts to a reference/validation value.
        //
        // Enabled by default.  Set ADAPTIVE_SKIP_MODULARITY=1 to skip.
        double modularity_estimate = 0.0;
        {
            const char* skip_mod = std::getenv("ADAPTIVE_SKIP_MODULARITY");
            if (skip_mod && std::string(skip_mod) == "1") {
                // Use the same heuristic as C++ runtime scoreBase
                modularity_estimate = std::min(0.9, clustering_coeff * 1.5);
            } else {
                modularity_estimate = computeFastModularity<uint32_t>(g, 1.0, 3);
            }
        }
        
        // ============================================================
        // 4. COMMUNITY COUNT (fast Leiden for estimate)
        // ============================================================
        int community_count = 1;  // Default: 1 large component
        
        // For community count, we can use a rough estimate based on graph density
        // More accurate counting is done during AdaptiveOrder when needed
        double density = static_cast<double>(num_edges) / (static_cast<double>(num_nodes) * (num_nodes - 1));
        if (density < 0.001 && num_nodes > 1000) {
            // Sparse graph likely has multiple communities
            // Rough estimate based on graph structure
            community_count = static_cast<int>(std::sqrt(num_nodes / 100.0)) + 1;
        } else if (hub_concentration > 0.5) {
            // Hub-dominated graph
            community_count = static_cast<int>(hub_concentration * 10) + 1;
        }
        
        t.Stop();
        
        // ============================================================
        // OUTPUT (format for Python parsing)
        // ============================================================
        std::cout << "=== Graph Topology Features ===" << std::endl;
        PrintTime("Clustering Coefficient", clustering_coeff);
        PrintTime("Avg Path Length", avg_path_length);
        PrintTime("Diameter Estimate", diameter_estimate);
        PrintTime("Community Count Estimate", community_count);
        PrintTime("Degree Variance", degree_variance);
        PrintTime("Hub Concentration", hub_concentration);
        PrintTime("Avg Degree", avg_degree);
        PrintTime("Graph Density", density);
        PrintTime("Packing Factor", packing_factor);
        PrintTime("Packing Factor CL", packing_factor_cl);
        PrintTime("Forward Edge Fraction", forward_edge_fraction);
        PrintTime("Working Set Ratio", working_set_ratio);
        PrintTime("Vertex Significance Skewness", vertex_significance_skewness);
        PrintTime("Window Neighbor Overlap", window_neighbor_overlap);
        PrintTime("Sampled Locality Score", sampled_locality_score);
        PrintTime("Modularity", modularity_estimate);
        PrintTime("Topology Analysis Time", t.Seconds());
        std::cout << "===============================" << std::endl;

        // ---- Self-recording: save graph properties to database ----
        if (graphbrew::database::SelfRecordingEnabled()) {
            graphbrew::database::GraphProperties props;
            props.graph_name           = GetGraphNameHint();
            props.nodes                = num_nodes;
            props.edges                = num_edges;
            props.modularity           = modularity_estimate;
            props.degree_variance      = degree_variance;
            props.hub_concentration    = hub_concentration;
            props.clustering_coeff     = clustering_coeff;
            props.avg_degree           = avg_degree;
            props.avg_path_length      = avg_path_length;
            props.diameter_estimate    = diameter_estimate;
            props.community_count      = community_count;
            props.packing_factor       = packing_factor;
            props.packing_factor_cl    = packing_factor_cl;
            props.forward_edge_fraction = forward_edge_fraction;
            props.working_set_ratio    = working_set_ratio;
            props.wsr_l1               = deg_features.wsr_l1;
            props.wsr_l2               = deg_features.wsr_l2;
            props.density              = density;
            props.vertex_significance_skewness = vertex_significance_skewness;
            props.window_neighbor_overlap = window_neighbor_overlap;
            props.sampled_locality_score = sampled_locality_score;

            graphbrew::database::BenchmarkDatabase::Get().update_graph_props(props);
        }
    }

    /**
     * Perceptron-style Algorithm Selector
     * 
     * Uses learned weights from multi-algorithm correlation analysis to
     * predict the best reordering algorithm based on community features.
     * 
     * The perceptron computes a score for each candidate algorithm:
     * 
     *   score = bias 
     *         + w_modularity * modularity 
     *         + w_log_nodes * log10(nodes)
     *         + w_log_edges * log10(edges)
     *         + w_density * density
     *         + w_avg_degree * avg_degree / 100
     *         + w_degree_variance * degree_variance
     *         + w_hub_concentration * hub_concentration
     *         + w_clustering_coeff * clustering_coeff      (NEW)
     *         + w_avg_path_length * avg_path_length / 10   (NEW)
     *         + w_diameter * diameter / 50                 (NEW)
     *         + w_community_count * log10(community_count) (NEW)
     *         + w_reorder_time * reorder_time              (NEW)
     * 
     * The score is then multiplied by benchmark_weights[current_benchmark]
     * to adjust for benchmark-specific performance characteristics.
     * 
     * The algorithm with the highest score is selected.
     * 
     * Weights were learned from benchmarking multiple graph algorithms
     * (BFS, PR, SSSP, BC, CC) across graphs with varying modularity.
     * 
     * WEIGHT CATEGORIES:
     * - Core weights: bias, w_modularity, w_log_nodes, w_log_edges, w_density,
     *                 w_avg_degree, w_degree_variance, w_hub_concentration
     * - Extended graph structure: w_clustering_coeff, w_avg_path_length, 
     *                             w_diameter, w_community_count
     * - Cache impact: cache_l1_impact, cache_l2_impact, cache_l3_impact,
     *                 cache_dram_penalty (used during training to adjust bias)
     * - Reorder time: w_reorder_time (penalty for slow reordering)
     * - Benchmark-specific: benchmark_weights[pr|bfs|cc|sssp|bc] (multiplier per benchmark)
     * 
     * WEIGHT LOADING: Weights can be loaded from a JSON file at runtime.
     * If the file exists, it overrides the hardcoded defaults.
     * Environment var: PERCEPTRON_WEIGHTS_FILE can override the default path.
     * 
     * Format: {"AlgorithmName": {"bias": X, "w_modularity": X, ...}, ...}
     * 
     * AUTO-CLUSTERING TYPE SYSTEM (DEPRECATED — see reorder_database.h):
     * Weights are now loaded from results/data/adaptive_models.json via
     * LoadPerceptronWeightsFromDB() in reorder_database.h.  C++ trains
     * perceptron, DT, and hybrid models at runtime from benchmarks.json +
     * graph_properties.json.  The legacy type_N/ directory structure is
     * no longer generated.
     *
     * At runtime, the system:
     * 1. Queries the streaming database for oracle/kNN predictions
     * 2. Falls back to adaptive_models.json if the database has <3 graphs
     * 3. Falls back to hardcoded defaults if no model file exists
     */
    
    /**
     * Graph type enum for graph-type-specific weight selection
     * 
     * Different graph types have different structural properties that
     * benefit from different reordering strategies:
     * 
     * - SOCIAL: High modularity, community structure, power-law degrees
     *           Best: GraphBrew, RabbitOrder
     * - ROAD: Mesh-like, low modularity, planar structure
     *         Best: RCMOrder (bandwidth reduction)
     * - WEB: High hub concentration, bow-tie structure
     *        Best: HubClusterDBG, HubSort
     * - POWERLAW: RMAT-like, highly skewed degree distribution
     *             Best: RabbitOrder, HubCluster
     * - UNIFORM: Random graphs, uniform degree distribution
     *            Best: Original or light reordering (DBG)
     * - GENERIC: Unknown or mixed - use default weights
     * 
     * GraphType enum is defined in reorder/reorder_types.h
     * Import into class scope for backward compatibility.
     */
    using GraphType = ::GraphType;
    static constexpr GraphType GRAPH_GENERIC  = ::GRAPH_GENERIC;
    static constexpr GraphType GRAPH_SOCIAL   = ::GRAPH_SOCIAL;
    static constexpr GraphType GRAPH_ROAD     = ::GRAPH_ROAD;
    static constexpr GraphType GRAPH_WEB      = ::GRAPH_WEB;
    static constexpr GraphType GRAPH_POWERLAW = ::GRAPH_POWERLAW;
    static constexpr GraphType GRAPH_UNIFORM  = ::GRAPH_UNIFORM;
    
    /**
     * Convert graph type enum to string (delegates to global function)
     */
    static std::string GraphTypeToString(GraphType type) {
        return ::GraphTypeToString(type);
    }
    
    /**
     * Convert string to graph type enum (delegates to global function)
     */
    static GraphType GetGraphType(const std::string& name) {
        return ::GetGraphType(name);
    }
    
    /**
     * Auto-detect graph type from graph features (delegates to global function)
     */
    static GraphType DetectGraphType(double modularity, double degree_variance, 
                                      double hub_concentration, double avg_degree,
                                      size_t num_nodes) {
        return ::DetectGraphType(modularity, degree_variance, hub_concentration, 
                                 avg_degree, num_nodes);
    }
    
    // BenchmarkType is now defined in reorder/reorder_types.h
    // Alias for backward compatibility within BuilderBase
    using BenchmarkType = ::BenchmarkType;
    
    // Use global GetBenchmarkType function
    static BenchmarkType GetBenchmarkType(const std::string& name) {
        return ::GetBenchmarkType(name);
    }
    
    /**
     * Selection mode for AdaptiveOrder algorithm selection
     * 
     * SelectionMode enum is defined in reorder/reorder_types.h
     * Import into class scope for backward compatibility.
     */
    using SelectionMode = ::SelectionMode;
    static constexpr SelectionMode MODE_FASTEST_REORDER    = ::MODE_FASTEST_REORDER;
    static constexpr SelectionMode MODE_FASTEST_EXECUTION  = ::MODE_FASTEST_EXECUTION;
    static constexpr SelectionMode MODE_BEST_ENDTOEND      = ::MODE_BEST_ENDTOEND;
    static constexpr SelectionMode MODE_BEST_AMORTIZATION  = ::MODE_BEST_AMORTIZATION;
    
    /**
     * Convert selection mode to string (delegates to global function)
     */
    static std::string SelectionModeToString(SelectionMode mode) {
        return ::SelectionModeToString(mode);
    }
    
    /**
     * Convert string to selection mode (delegates to global function)
     */
    static SelectionMode GetSelectionMode(const std::string& name) {
        return ::GetSelectionMode(name);
    }
    
    // PerceptronWeights is now defined in reorder/reorder_types.h
    // Alias for backward compatibility within BuilderBase
    using PerceptronWeights = ::PerceptronWeights;

    /**
     * Get default perceptron weights for all reordering algorithms.
     * 
     * Delegates to the global ::GetPerceptronWeights() function in reorder_types.h.
     * This ensures a single source of truth for default weights across the codebase.
     */
    static const std::map<std::string, PerceptronWeights>& GetPerceptronWeights() {
        return ::GetPerceptronWeights();
    }

    /**
     * Parse weights from JSON (delegates to global function)
     */
    static bool ParseWeightsFromJSON(const std::string& json_content,
                                  std::map<std::string, PerceptronWeights>& weights) {
        return ::ParseWeightsFromJSON(json_content, weights);
    }

    /**
     * Select algorithm with fastest reorder time based on w_reorder_time weight.
     * Delegates to the global ::SelectFastestReorderFromWeights in reorder_types.h.
     */
    static PerceptronSelection SelectFastestReorderFromWeights(
        const std::map<std::string, PerceptronWeights>& weights, bool verbose = false) {
        return ::SelectFastestReorderFromWeights(weights, verbose);
    }
    
    // Threshold constant - defined in reorder/reorder_types.h
    static constexpr double UNKNOWN_TYPE_DISTANCE_THRESHOLD = ::UNKNOWN_TYPE_DISTANCE_THRESHOLD;
    
    /**
     * Check if a graph is far from known types (delegates to global function)
     */
    static bool IsDistantGraphType(double type_distance, double type_radius = 0.0) {
        return ::IsDistantGraphType(type_distance, type_radius);
    }
    
    /**
     * Load perceptron weights using graph features to find the best type match.
     * Delegates to the global ::LoadPerceptronWeightsForFeatures in reorder_types.h.
     */
    static std::map<std::string, PerceptronWeights> LoadPerceptronWeightsForFeatures(
        double modularity, double degree_variance, double hub_concentration,
        double avg_degree, size_t num_nodes, size_t num_edges, bool verbose = false,
        double clustering_coeff = 0.0) {
        return ::LoadPerceptronWeightsForFeatures(modularity, degree_variance, hub_concentration,
                                                   avg_degree, num_nodes, num_edges, verbose,
                                                   clustering_coeff);
    }
    
    /**
     * Select best reordering algorithm using feature-based type matching.
     * Delegates to the global ::SelectReorderingPerceptronWithFeatures in reorder_types.h.
     */
    PerceptronSelection SelectReorderingPerceptronWithFeatures(
        const CommunityFeatures& feat,
        double global_modularity, double global_degree_variance,
        double global_hub_concentration, size_t num_nodes, size_t num_edges,
        BenchmarkType bench = BENCH_GENERIC) {
        return ::SelectReorderingPerceptronWithFeatures(feat, global_modularity, global_degree_variance,
                                                         global_hub_concentration, num_nodes, num_edges, bench);
    }
    
    /**
     * Select best reordering algorithm with MODE-AWARE selection.
     * Delegates to the global ::SelectReorderingWithMode in reorder_types.h.
     */
    PerceptronSelection SelectReorderingWithMode(
        const CommunityFeatures& feat,
        double global_modularity, double global_degree_variance,
        double global_hub_concentration, size_t num_nodes, size_t num_edges,
        SelectionMode mode, const std::string& graph_name = "",
        BenchmarkType bench = BENCH_GENERIC, bool verbose = false) {
        return ::SelectReorderingWithMode(feat, global_modularity, global_degree_variance,
                                           global_hub_concentration, num_nodes, num_edges,
                                           mode, graph_name, bench, verbose);
    }

    CommunityFeatures ComputeCommunityFeatures(
        const std::vector<NodeID_>& comm_nodes,
        const CSRGraph<NodeID_, DestID_, invert>& g,
        const std::unordered_set<NodeID_>& node_set,
        bool compute_extended = true)
    {
        // Delegate to standalone function in reorder_types.h
        return ::ComputeCommunityFeaturesStandalone<NodeID_, DestID_, invert>(
            comm_nodes, g, node_set, compute_extended);
    }

    /**
    /**
     * Compute dynamic minimum community size threshold.
     * Delegates to the global ::ComputeDynamicMinCommunitySize in reorder_types.h.
     */
    static size_t ComputeDynamicMinCommunitySize(size_t num_nodes, 
                                                  size_t num_communities,
                                                  size_t avg_community_size = 0) {
        return ::ComputeDynamicMinCommunitySize(num_nodes, num_communities, avg_community_size);
    }
    
    /**
     * Compute dynamic threshold for when to apply local reordering.
     * Delegates to ::ComputeDynamicLocalReorderThreshold in reorder_types.h
     */
    static size_t ComputeDynamicLocalReorderThreshold(size_t num_nodes,
                                                       size_t num_communities,
                                                       size_t avg_community_size = 0) {
        return ::ComputeDynamicLocalReorderThreshold(num_nodes, num_communities, avg_community_size);
    }

    /**
     * Select best reordering algorithm based on community features.
     * Delegates to the global ::SelectBestReorderingForCommunity in reorder_types.h.
     */
    PerceptronSelection SelectBestReorderingForCommunity(CommunityFeatures feat, 
                                                     double global_modularity,
                                                     double global_degree_variance,
                                                     double global_hub_concentration,
                                                     double global_avg_degree,
                                                     size_t num_nodes, size_t num_edges,
                                                     BenchmarkType bench = BENCH_GENERIC,
                                                     GraphType graph_type = GRAPH_GENERIC,
                                                     SelectionMode mode = MODE_FASTEST_EXECUTION,
                                                     const std::string& graph_name = "",
                                                     size_t dynamic_min_size = 0) {
        return ::SelectBestReorderingForCommunity(feat, global_modularity, global_degree_variance,
                                                   global_hub_concentration, global_avg_degree,
                                                   num_nodes, num_edges, bench, graph_type,
                                                   mode, graph_name, dynamic_min_size);
    }

    // ========================================================================
    // ADAPTIVE REORDERING - Delegates to standalone implementations
    // ========================================================================
    // See reorder_adaptive.h for full implementations using GVE-Leiden (native CSR).
    // These delegates maintain backward compatibility with existing code.
    
    /**
     * Main entry point for Adaptive reordering - delegates to standalone.
     * Format: -o 14[:_[:_[:_[:selection_mode[:graph_name]]]]]
     *   Positions 0-2: reserved (unused)
     *   Position 3: selection_mode (0-3)
     *   Position 4: graph_name (string)
     */
    void GenerateAdaptiveMapping(CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids, bool useOutdeg,
                                 std::vector<std::string> reordering_options) {
        ::GenerateAdaptiveMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options);
    }
    
    /**
     * Full-graph adaptive mode - delegates to standalone.
     */
    void GenerateAdaptiveMappingFullGraph(CSRGraph<NodeID_, DestID_, invert> &g,
                                          pvector<NodeID_> &new_ids, bool useOutdeg,
                                          std::vector<std::string> reordering_options) {
        ::GenerateAdaptiveMappingFullGraphStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options);
    }
    
    /**
     * Recursive per-community adaptive mode - delegates to standalone.
     */
    void GenerateAdaptiveMappingRecursive(
        const CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids, 
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int depth = 0,
        bool verbose = true,
        SelectionMode selection_mode = MODE_FASTEST_EXECUTION,
        const std::string& graph_name = "") {
        ::GenerateAdaptiveMappingRecursiveStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options, depth, verbose, selection_mode, graph_name);
    }


    //==========================================================================
    // GRAPHBREW UNIFIED ENTRY POINT
    //
    // Extended GraphBrewOrder with configurable clustering and final reordering.
    //
    // Presets (expand to tokens):
    //   leiden      - GVE-CSR Leiden community detection (default)
    //   rabbit      - Full RabbitOrder pipeline (single-pass, no Leiden)
    //   hubcluster  - Leiden + hub-cluster native ordering
    //
    // Token mode (all args are parseGraphBrewConfig tokens):
    //   hrab, dfs, bfs, conn, dbg, corder, etc.
    //
    // Examples:
    //   -o 12                      # Default: leiden preset
    //   -o 12:leiden:6             # Leiden, HubSortDBG per-community
    //   -o 12:leiden:8:1.0:3      # Leiden, RabbitOrder, resolution 1.0, 3 passes
    //   -o 12:hrab:gvecsr:0.75    # Token mode: hybrid + GVE-CSR + resolution
    //==========================================================================
    
    // ========================================================================
    // GRAPHBREW REORDERING - Powered by GraphBrew pipeline
    // ========================================================================
    //
    // GraphBrewOrder (ID 12) uses GraphBrew's Leiden community detection with
    // GVE-CSR aggregation, then applies any reordering algorithm (0-11) per
    // community. Supports recursive depth (auto or explicit), adaptive
    // sub-algorithm selection, hub extraction, and small-community merging.
    //
    // All parsing routes through graphbrew::parseGraphBrewConfig() (one parser).
    // Named presets expand to equivalent tokens:
    //
    //   Preset        Token expansion
    //   ──────        ───────────────
    //   leiden        gvecsr totalm refine0 graphbrew
    //   rabbit        rabbitorder 0.5
    //   hubcluster    hubcluster
    //
    // After a preset, remaining args are positional:
    //   -o 12:leiden[:final_algo[:resolution[:passes[:depth[:sub]]]]]
    //
    // Without a preset, all args are tokens for parseGraphBrewConfig:
    //   -o 12:hrab:gvecsr:0.75
    // ========================================================================
    
    /**
     * Parse GraphBrew CLI options and build a GraphBrewConfig.
     * 
     * All parsing is routed through graphbrew::parseGraphBrewConfig() in
     * reorder_graphbrew.h.  Named presets (leiden, rabbit, hubcluster)
     * are expanded to equivalent tokens before being fed to that parser,
     * so there is exactly ONE token parser for the whole system.
     * 
     * Formats:
     *   -o 12                                     # Default (leiden preset)
     *   -o 12:leiden:8:0.75:3                     # Preset + positional overrides
     *   -o 12:hrab:gvecsr:0.75                    # Direct token mode
     */
    graphbrew::GraphBrewConfig ParseGraphBrewConfig(
        const std::vector<std::string>& options,
        double auto_resolution) {
        
        // ── Default (no options) → leiden preset ────────────────────────
        if (options.empty() || options[0].empty()) {
            graphbrew::GraphBrewConfig config = graphbrew::parseGraphBrewConfig(
                {"gvecsr", "totalm", "refine0", "graphbrew"});
            config.resolution = auto_resolution;
            return config;
        }
        
        const std::string& first = options[0];
        
        // ── Preset expansion table ──────────────────────────────────────
        // Each preset maps to tokens understood by parseGraphBrewConfig.
        // "graphbrew" token → LAYER ordering + finalAlgoId=8 + smallCommunityMerging
        struct PresetDef {
            std::vector<std::string> tokens;
        };
        static const std::map<std::string, PresetDef> PRESETS = {
            {"leiden",      {{"gvecsr", "totalm", "refine0", "graphbrew"}}},
            {"rabbit",      {{"rabbitorder", "0.5"}}},
            {"hubcluster",  {{"hubcluster"}}},
        };
        
        auto preset_it = PRESETS.find(first);
        graphbrew::GraphBrewConfig config;
        
        if (preset_it != PRESETS.end()) {
            // ── Known preset → expand tokens, then parse positional tail ─
            config = graphbrew::parseGraphBrewConfig(preset_it->second.tokens);
            
            // Apply LAYER-mode defaults (only if the preset didn't set another ordering)
            if (config.ordering == graphbrew::OrderingStrategy::CONNECTIVITY_BFS) {
                config.ordering = graphbrew::OrderingStrategy::LAYER;
            }
            if (config.ordering == graphbrew::OrderingStrategy::LAYER) {
                config.useSmallCommunityMerging = true;
                if (config.finalAlgoId < 0) config.finalAlgoId = 8;
            }
            
            // Apply auto-resolution when the preset/token didn't set an explicit one
            if (config.resolution == reorder::DEFAULT_RESOLUTION) {
                config.resolution = auto_resolution;
            }
            
            // Positional overrides: final_algo, resolution, passes, depth, sub
            if (options.size() > 1 && !options[1].empty()) {
                try { config.finalAlgoId = std::stoi(options[1]); } catch (...) {}
            }
            if (options.size() > 2 && !options[2].empty()) {
                const std::string& res = options[2];
                if (res == "auto" || res == "0") {
                    // keep auto_resolution
                } else if (res.rfind("dynamic", 0) == 0) {
                    config.useDynamicResolution = true;
                } else {
                    try {
                        double r = std::stod(res);
                        if (r > 0 && r <= 3) config.resolution = r;
                    } catch (...) {}
                }
            }
            if (options.size() > 3 && !options[3].empty()) {
                try {
                    int passes = std::stoi(options[3]);
                    if (passes > 0 && passes <= 50) config.maxPasses = passes;
                } catch (...) {}
            }
            if (options.size() > 4 && !options[4].empty()) {
                const std::string& depthStr = options[4];
                if (depthStr == "recursive" || depthStr == "recurse") {
                    config.recursiveDepth = std::max(config.recursiveDepth, 1);
                } else {
                    try {
                        int d = std::stoi(depthStr);
                        if (d >= 0 && d <= 10) config.recursiveDepth = d;
                    } catch (...) {}
                }
            }
            if (options.size() > 5 && !options[5].empty()) {
                const std::string& subStr = options[5];
                if (subStr == "auto" || subStr == "adaptive") {
                    config.subAlgoId = -1;
                } else {
                    try {
                        int a = std::stoi(subStr);
                        if (a >= 0 && a <= 11) config.subAlgoId = a;
                    } catch (...) {}
                }
            }
            
            // Named-token fallback: non-numeric tail tokens parsed as named
            // options.  Allows e.g. "12:leiden:recursive" or "12:leiden:hubx"
            // where "recursive" / "hubx" land at positional slots but are
            // actually named tokens that parseGraphBrewConfig understands.
            for (size_t i = 1; i < options.size(); ++i) {
                const auto& tok = options[i];
                if (tok.empty()) continue;
                // Skip tokens already handled positionally (numbers, "auto", "dynamic", etc.)
                if (tok == "auto" || tok == "adaptive" || tok == "dynamic") continue;
                bool is_numeric = false;
                try { std::stod(tok); is_numeric = true; } catch (...) {}
                if (is_numeric) continue;
                // Parse as a named GraphBrew token and merge flags
                auto extra = graphbrew::parseGraphBrewConfig({tok});
                if (extra.recursiveDepth > 0 && config.recursiveDepth < extra.recursiveDepth)
                    config.recursiveDepth = extra.recursiveDepth;
                if (extra.recursiveDepth == 0 && (tok == "flat" || tok == "norecurse"))
                    config.recursiveDepth = 0;  // explicit flat override
                if (extra.useHubExtraction) config.useHubExtraction = true;
                if (extra.useGorderIntra) config.useGorderIntra = true;
                if (extra.useHubSort) config.useHubSort = true;
                if (extra.useRCMSuper) config.useRCMSuper = true;
                if (extra.useCommunityMerging) config.useCommunityMerging = true;
                // Named ordering overrides LAYER only when explicitly set
                if (extra.ordering != graphbrew::OrderingStrategy::CONNECTIVITY_BFS &&
                    extra.ordering != graphbrew::OrderingStrategy::LAYER) {
                    config.ordering = extra.ordering;
                }
                if (extra.hasExplicitOrdering) config.hasExplicitOrdering = true;
                // COMPOSE-axis tokens (sg_*, comm_*, intra_*, sgresN.N, cd_*).
                // Each compose token sets exactly one axis from default; merging
                // any non-default value from `extra` is safe because the parser
                // never sets these axes implicitly.
                if (extra.superGraphOrder != graphbrew::SuperGraphOrder::None)
                    config.superGraphOrder = extra.superGraphOrder;
                if (extra.communityOrder != graphbrew::CommunityOrder::SizeDesc)
                    config.communityOrder = extra.communityOrder;
                if (extra.intraCommunityOrder != graphbrew::IntraCommunityOrder::BFSFromHub)
                    config.intraCommunityOrder = extra.intraCommunityOrder;
                if (extra.refinementPass != graphbrew::RefinementPass::None)
                    config.refinementPass = extra.refinementPass;
                if (extra.superGraphResolution != 0.10)
                    config.superGraphResolution = extra.superGraphResolution;
                // cd_rabbit / cd_leiden after a preset must override the preset's CD
                if (tok == "cd_rabbit" || tok == "cd:rabbit" || tok == "cdrabbit" ||
                    tok == "rabbit" || tok == "rabbitorder")
                    config.algorithm = graphbrew::GraphBrewAlgorithm::RABBIT_ORDER;
                if (tok == "cd_leiden" || tok == "cd:leiden" || tok == "cdleiden")
                    config.algorithm = graphbrew::GraphBrewAlgorithm::LEIDEN;
                if (tok == "cd_parallel" || tok == "cd:parallel" ||
                    tok == "community_parallel")
                    config.deterministicCommunityDetection = false;
                if (tok == "cd_serial" || tok == "cd:serial" ||
                    tok == "community_serial")
                    config.deterministicCommunityDetection = true;
            }
        } else {
            // ── Direct token mode (hrab, dfs, conn, graphbrew:hrab, etc.) ─
            config = graphbrew::parseGraphBrewConfig(options);
            
            if (config.ordering == graphbrew::OrderingStrategy::CONNECTIVITY_BFS) {
                config.ordering = graphbrew::OrderingStrategy::LAYER;
            }
            if (config.ordering == graphbrew::OrderingStrategy::LAYER) {
                config.useSmallCommunityMerging = true;
                if (config.finalAlgoId < 0) config.finalAlgoId = 8;
            }
            if (config.resolution == reorder::DEFAULT_RESOLUTION) {
                config.resolution = auto_resolution;
            }
        }
        
        return config;
    }
    
    /**
     * Unified GraphBrew entry point - powered by GraphBrew pipeline.
     * 
     * Pipeline:
     * 1. Parse options → GraphBrewConfig (with GRAPHBREW ordering)
     * 2. Run GraphBrew community detection (Leiden or RabbitOrder)
     * 3. Classify communities into small/large
     * 4. Merge small communities, apply heuristic algo selection
     * 5. Apply final reordering algorithm per large community
     * 6. Compose final vertex permutation
     */
    void GenerateGraphBrewMappingUnified(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        bool useOutdeg,
        std::vector<std::string> reordering_options) {
        
        Timer totalTimer;
        totalTimer.Start();
        
        const int64_t N = g.num_nodes();
        const int64_t E = g.num_edges();
        
        // Parse options to GraphBrew config
        double auto_resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        graphbrew::GraphBrewConfig config = ParseGraphBrewConfig(reordering_options, auto_resolution);
        
        ReorderingAlgo finalAlgo = static_cast<ReorderingAlgo>(
            (config.finalAlgoId >= 0 && config.finalAlgoId <= 11) 
            ? config.finalAlgoId : 8);
        
        printf("GraphBrew: finalAlgo=%s (%d), resolution=%.4f, maxPasses=%d, depth=%s, subAlgo=%s\n",
               ReorderingAlgoStr(finalAlgo).c_str(), config.finalAlgoId, config.resolution,
               config.maxPasses,
               config.recursiveDepth < 0 ? "auto" : std::to_string(config.recursiveDepth).c_str(),
               config.subAlgoId < 0 ? "auto" : ReorderingAlgoStr(static_cast<ReorderingAlgo>(config.subAlgoId)).c_str());
        
        // If hubcluster variant or no external dispatch needed, delegate directly to GraphBrew
        if (config.ordering != graphbrew::OrderingStrategy::LAYER) {
            printf("GraphBrew: delegating to GraphBrew native ordering\n");
            new_ids.resize(N);
            graphbrew::generateGraphBrewMapping<K>(g, new_ids, config);
            totalTimer.Stop();
            PrintTime("GraphBrew Total Time", totalTimer.Seconds());
            // Stage config-level metadata for GenerateMapping's ReorderMeta hint
            {
                using namespace graphbrew::database;
                auto& staged = GetStagedReorderMeta();
                staged.resolution = config.resolution;
                staged.final_algo = ReorderingAlgoStr(finalAlgo);
                staged.depth      = config.recursiveDepth;
            }
            return;
        }
        
        // If RabbitOrder algorithm, use GraphBrew's native RabbitOrder pipeline
        // with degree-sort preprocessing to match standalone RABBITORDER_csr quality.
        // Without physical CSR relabeling, community detection accesses scattered
        // memory locations which degrades BFS ordering quality by 5-30x.
        if (config.algorithm == graphbrew::GraphBrewAlgorithm::RABBIT_ORDER) {
            printf("GraphBrew: using GraphBrew RabbitOrder pipeline (with degree-sort preprocessing)\n");
            
            // Step 1: Degree-sort preprocessing (same as RABBITORDER_csr)
            pvector<NodeID_> sort_ids(N, -1);
            GenerateSortMappingRabbit(g, sort_ids, true, true);
            CSRGraph<NodeID_, DestID_, invert> g_sorted = RelabelByMapping(g, sort_ids);
            
            // Step 2: Run GraphBrew RabbitOrder on degree-sorted graph
            pvector<NodeID_> rabbit_ids(N);
            graphbrew::generateGraphBrewMapping<K>(g_sorted, rabbit_ids, config);
            
            // Step 3: Compose permutations: new_ids[orig] = rabbit_ids[sort_ids[orig]]
            new_ids.resize(N);
            #pragma omp parallel for
            for (int64_t n = 0; n < N; n++) {
                new_ids[n] = rabbit_ids[sort_ids[n]];
            }
            
            totalTimer.Stop();
            PrintTime("GraphBrew Total Time", totalTimer.Seconds());
            // Stage config-level metadata for GenerateMapping's ReorderMeta hint
            {
                using namespace graphbrew::database;
                auto& staged = GetStagedReorderMeta();
                staged.resolution = config.resolution;
                staged.final_algo = "RabbitOrder";
            }
            return;
        }
        
        // ===== Phase 1: Community Detection via GraphBrew =====
        Timer detectTimer;
        detectTimer.Start();
        
        auto result = graphbrew::runGraphBrew<K>(g, config);
        
        detectTimer.Stop();
        printf("GraphBrew detection: %d passes, %d iters, %zu communities, %.4fs\n",
               result.totalPasses, result.totalIterations, result.numCommunities,
               detectTimer.Seconds());
        
        // Community merging (optional, matches old GraphBrew behavior)
        if (config.useCommunityMerging) {
            Timer mergeTimer;
            mergeTimer.Start();
            size_t finalComms = graphbrew::mergeCommunities<K>(result.membership, g, config.targetCommunities);
            mergeTimer.Stop();
            result.numCommunities = finalComms;
            printf("GraphBrew: community merge: %zu communities, %.4fs\n", finalComms, mergeTimer.Seconds());
        }

        // Stage reorder metadata for GenerateMapping's ReorderMeta hint
        {
            using namespace graphbrew::database;
            auto& staged = GetStagedReorderMeta();
            staged.num_communities = static_cast<int>(result.numCommunities);
            staged.modularity      = result.modularity;
            staged.num_passes      = result.totalPasses;
            staged.resolution      = config.resolution;
            staged.final_algo      = ReorderingAlgoStr(static_cast<ReorderingAlgo>(
                (config.finalAlgoId >= 0 && config.finalAlgoId <= 11) ? config.finalAlgoId : 8));
            staged.depth           = config.recursiveDepth;
            if (config.subAlgoId >= 0 && config.subAlgoId <= 11) {
                staged.sub_algo = ReorderingAlgoStr(static_cast<ReorderingAlgo>(config.subAlgoId));
            }
        }
        
        // ===== Phase 2: Classify Communities =====
        const auto& membership = result.membership;
        
        // Count community sizes
        std::unordered_map<K, size_t> comm_sizes;
        for (int64_t v = 0; v < N; ++v) {
            comm_sizes[membership[v]]++;
        }
        
        // Compute dynamic threshold (same formula as old GraphBrew)
        size_t non_empty = comm_sizes.size();
        size_t avg_comm_size = (non_empty > 0) ? static_cast<size_t>(N) / non_empty : N;
        
        size_t min_community_size = config.smallCommunityThreshold;
        if (min_community_size == 0) {
            // Dynamic: max(50, min(avg/4, sqrt(N))), capped at 2000
            min_community_size = ComputeDynamicMinCommunitySize(
                static_cast<size_t>(N), non_empty, avg_comm_size);
        }
        
        printf("GraphBrew: threshold=%zu (avg_comm=%zu, num_comm=%zu)\n",
               min_community_size, avg_comm_size, non_empty);
        
        // Separate communities into small and large
        std::unordered_set<K> small_comms, large_comms;
        for (auto& [comm_id, size] : comm_sizes) {
            if (size < min_community_size) {
                small_comms.insert(comm_id);
            } else {
                large_comms.insert(comm_id);
            }
        }
        
        // Collect nodes per community
        std::unordered_map<K, std::vector<NodeID_>> comm_nodes;
        for (int64_t v = 0; v < N; ++v) {
            comm_nodes[membership[v]].push_back(static_cast<NodeID_>(v));
        }
        
        // Sort large communities by size (descending) for deterministic ordering
        std::vector<std::pair<size_t, K>> sorted_large_comms;
        for (K c : large_comms) {
            sorted_large_comms.push_back({comm_sizes[c], c});
        }
        std::sort(sorted_large_comms.begin(), sorted_large_comms.end(), std::greater<>());
        
        printf("GraphBrew: %zu large, %zu small communities\n",
               large_comms.size(), small_comms.size());
        
        // ===== Phase 3: Assign IDs =====
        new_ids.resize(N);
        std::fill(new_ids.begin(), new_ids.end(), static_cast<NodeID_>(-1));
        NodeID_ current_id = 0;
        
        // 3a. Handle small communities (merge and apply heuristic)
        if (!small_comms.empty()) {
            std::vector<NodeID_> small_nodes;
            std::unordered_set<NodeID_> small_node_set;
            for (K c : small_comms) {
                for (NodeID_ v : comm_nodes[c]) {
                    small_nodes.push_back(v);
                    small_node_set.insert(v);
                }
            }
            
            if (config.useSmallCommunityMerging && small_nodes.size() >= 100) {
                // Compute features for merged small communities and select algorithm
                auto merged_feat = ComputeMergedCommunityFeatures(g, small_nodes, small_node_set);
                ReorderingAlgo small_algo = SelectAlgorithmForSmallGroup(merged_feat);
                
                printf("GraphBrew: %zu small-community nodes -> %s\n",
                       small_nodes.size(), ReorderingAlgoStr(small_algo).c_str());
                
                ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                    g, small_nodes, small_node_set, small_algo, useOutdeg,
                    new_ids, current_id);
            } else {
                // Simple degree-sorted assignment for tiny groups
                std::vector<std::pair<int64_t, NodeID_>> degree_nodes;
                degree_nodes.reserve(small_nodes.size());
                for (NodeID_ v : small_nodes) {
                    int64_t deg = useOutdeg ? g.out_degree(v) : g.in_degree(v);
                    degree_nodes.push_back({-deg, v});
                }
                std::sort(degree_nodes.begin(), degree_nodes.end());
                for (auto& [neg_deg, v] : degree_nodes) {
                    new_ids[v] = current_id++;
                }
            }
        }
        
        // ── Cache capacity detection (shared by auto-depth + recursive mode) ──
        const size_t llc_bytes = GetLLCSizeBytes();
        const size_t l2_bytes = std::max(size_t(256 * 1024),
            static_cast<size_t>(sysconf(_SC_LEVEL2_CACHE_SIZE)));
        // Working set per node: CSR offsets (4B) + neighbor array (avg_deg*4B)
        //   + algorithm arrays (2×8B for PR, BFS dist, etc.) + cache-line overhead
        // Use 2× safety factor to account for irregular access patterns and
        // cache-line waste on power-law degree distributions.
        const double avg_deg = static_cast<double>(E) / std::max(int64_t(1), N);
        const size_t bytes_per_node = static_cast<size_t>(
            avg_deg * sizeof(NodeID_) + 5 * sizeof(NodeID_) + 2 * sizeof(double));
        const size_t l2_node_capacity = l2_bytes / std::max(size_t(1), bytes_per_node);
        const size_t llc_node_capacity = llc_bytes / std::max(size_t(1), bytes_per_node);
        
        // ── Auto-depth: resolve -1 → 0 (flat) or 1 (recursive) ──
        // Recurse when the largest community exceeds half the effective LLC
        // capacity.  Even if the community technically fits LLC, sub-dividing
        // lets sub-communities fit L2, which yields better locality for the
        // per-element kernel (PR, BFS, etc.).  The ½ LLC threshold is a
        // conservative trigger derived from empirical validation: as-Skitter
        // shows 12.6% PR speedup from recursion at 338K vs 381K LLC capacity.
        if (config.recursiveDepth < 0) {
            size_t max_comm_size = 0;
            for (auto& [size, comm_id] : sorted_large_comms) {
                max_comm_size = std::max(max_comm_size, size);
            }
            const size_t auto_depth_threshold = llc_node_capacity / 2;
            if (max_comm_size > auto_depth_threshold) {
                config.recursiveDepth = 1;
                printf("GraphBrew: auto-depth=1 (max community %zu > LLC/2 threshold %zu nodes)\n",
                       max_comm_size, auto_depth_threshold);
            } else {
                config.recursiveDepth = 0;
                printf("GraphBrew: auto-depth=0 (all communities fit LLC/2, max=%zu <= %zu)\n",
                       max_comm_size, auto_depth_threshold);
            }
        }
        
        // 3b. Handle large communities
        Timer perCommTimer;
        perCommTimer.Start();
        
        if (config.recursiveDepth > 0) {
            // ===== RECURSIVE MODE: sub-divide each large community =====
            printf("GraphBrew: recursive depth=%d, sub-dividing %zu large communities\n",
                   config.recursiveDepth, sorted_large_comms.size());
            
            // Determine sub-community algorithm mode
            const bool autoSubAlgo = (config.subAlgoId < 0);
            const ReorderingAlgo fixedSubAlgo = autoSubAlgo ? finalAlgo :
                static_cast<ReorderingAlgo>(config.subAlgoId);
            
            // Aggregated cache-fit statistics
            size_t total_sub_comms = 0, fits_l2 = 0, fits_llc = 0;
            size_t total_cross_edges = 0, total_internal_edges = 0;
            
            // Phase timing accumulators
            double time_edge_analysis = 0, time_sub_reorder = 0;
            std::unordered_map<int, size_t> algo_distribution;
            
            // Helper: reorder a sub-community using the already-built subgraph.
            // Extracts edges from sub_g (not global graph g) to avoid redundant
            // edge scanning and hash lookups — the key bottleneck fix.
            auto reorderFromLocalSubgraph = [&](
                const std::vector<NodeID_>& sc_local_nodes,
                const CSRGraph<NodeID_, DestID_, invert>& sub_g,
                const std::vector<NodeID_>& l2g,
                ReorderingAlgo algo,
                std::vector<NodeID_>& sc_map) {
                
                const size_t sc_size = sc_local_nodes.size();
                if (sc_size == 0) return;
                const size_t sg_N = static_cast<size_t>(sub_g.num_nodes());
                
                // Map sub_g local IDs → sub-sub sequential IDs (reuse sc_map)
                std::vector<NodeID_> ss2g(sc_size);
                for (size_t i = 0; i < sc_size; ++i) {
                    NodeID_ lv = sc_local_nodes[i];
                    if (static_cast<size_t>(lv) < sg_N) sc_map[lv] = static_cast<NodeID_>(i);
                    ss2g[i] = (static_cast<size_t>(lv) < l2g.size()) ? l2g[lv] : lv;
                }
                
                // Extract edges from sub_g (much smaller than global graph)
                std::vector<std::pair<NodeID_, DestID_>> ss_edges;
                for (NodeID_ lv : sc_local_nodes) {
                    if (static_cast<size_t>(lv) >= sg_N) continue;
                    NodeID_ ss_src = sc_map[lv];
                    for (DestID_ neighbor : sub_g.out_neigh(lv)) {
                        NodeID_ nb = static_cast<NodeID_>(neighbor);
                        if (static_cast<size_t>(nb) < sg_N && sc_map[nb] != static_cast<NodeID_>(-1)) {
                            ss_edges.push_back({ss_src, static_cast<DestID_>(sc_map[nb])});
                        }
                    }
                }
                
                // Cleanup sc_map entries for reuse by next sub-community
                for (NodeID_ lv : sc_local_nodes) {
                    if (static_cast<size_t>(lv) < sg_N) sc_map[lv] = static_cast<NodeID_>(-1);
                }
                
                if (ss_edges.empty()) {
                    for (size_t i = 0; i < sc_size; ++i) new_ids[ss2g[i]] = current_id++;
                    return;
                }
                
                auto ss_g = MakeLocalGraphFromELStandalone<NodeID_, DestID_, invert>(ss_edges, false);
                const size_t ss_N = static_cast<size_t>(ss_g.num_nodes());
                pvector<NodeID_> ss_new_ids(ss_N, -1);
                ApplyBasicReorderingStandalone<NodeID_, DestID_, WeightT_, invert>(
                    ss_g, ss_new_ids, algo, useOutdeg);
                
                std::vector<NodeID_> reordered(sc_size);
                for (size_t i = 0; i < std::min(sc_size, ss_N); ++i) {
                    if (ss_new_ids[i] >= 0 && static_cast<size_t>(ss_new_ids[i]) < sc_size)
                        reordered[ss_new_ids[i]] = ss2g[i];
                    else
                        reordered[i] = ss2g[i];
                }
                for (size_t i = ss_N; i < sc_size; ++i) reordered[i] = ss2g[i];
                for (NodeID_ gv : reordered) new_ids[gv] = current_id++;
            };
            
            // Helper: select algorithm adaptively using sub_g degree stats.
            // Computes features directly from subgraph (no global graph scanning).
            auto selectSubAlgo = [&](const std::vector<NodeID_>& sc_local_nodes,
                                     const CSRGraph<NodeID_, DestID_, invert>& sub_g,
                                     size_t sc_size) -> ReorderingAlgo {
                if (!autoSubAlgo) return fixedSubAlgo;
                if (sc_size < 100) return finalAlgo;
                
                const size_t sg_N = static_cast<size_t>(sub_g.num_nodes());
                double sum_deg = 0, sum_deg_sq = 0, max_deg = 0;
                size_t valid_nodes = 0;
                for (NodeID_ lv : sc_local_nodes) {
                    if (static_cast<size_t>(lv) >= sg_N) continue;
                    double deg = static_cast<double>(sub_g.out_degree(lv));
                    sum_deg += deg;
                    sum_deg_sq += deg * deg;
                    if (deg > max_deg) max_deg = deg;
                    valid_nodes++;
                }
                if (valid_nodes == 0) return finalAlgo;
                
                double avg_deg = sum_deg / valid_nodes;
                double variance = (sum_deg_sq / valid_nodes) - (avg_deg * avg_deg);
                double degree_variance = (avg_deg > 0) ? std::sqrt(std::max(0.0, variance)) / avg_deg : 0;
                double hub_concentration = (sum_deg > 0) ? max_deg / sum_deg : 0;
                double density = (valid_nodes > 1) ?
                    sum_deg / (static_cast<double>(valid_nodes) * (valid_nodes - 1)) : 0;
                
                // Cache-tier-aware algorithm selection
                if (sc_size <= l2_node_capacity) {
                    // Fits in L2 — light touch
                    if (hub_concentration > 0.4) return HubSort;
                    return DBG;
                } else if (sc_size <= llc_node_capacity) {
                    // Fits in LLC — heavier restructuring worthwhile
                    if (degree_variance > 1.5 && hub_concentration > 0.5)
                        return HubClusterDBG;
                    if (hub_concentration > 0.3) return HubSort;
                    return RabbitOrder;
                } else {
                    // Exceeds LLC — maximize cache-line reuse
                    if (density > 0.01) return GOrder;
                    return RabbitOrder;
                }
            };
            
            for (auto& [size, comm_id] : sorted_large_comms) {
                auto& nodes = comm_nodes[comm_id];
                if (nodes.empty()) continue;
                
                std::unordered_set<NodeID_> node_set(nodes.begin(), nodes.end());
                
                // Skip subgraph construction for small communities
                if (nodes.size() < 200) {
                    ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                        g, nodes, node_set, finalAlgo, useOutdeg,
                        new_ids, current_id);
                    continue;
                }
                
                // Build induced subgraph edge list (directly in local IDs)
                std::unordered_map<NodeID_, NodeID_> g2l;
                std::vector<NodeID_> l2g(nodes.size());
                for (size_t i = 0; i < nodes.size(); ++i) {
                    g2l[nodes[i]] = static_cast<NodeID_>(i);
                    l2g[i] = nodes[i];
                }
                
                std::vector<std::pair<NodeID_, DestID_>> local_edges;
                for (NodeID_ node : nodes) {
                    NodeID_ local_src = g2l[node];
                    for (DestID_ neighbor : g.out_neigh(node)) {
                        NodeID_ dest = static_cast<NodeID_>(neighbor);
                        if (node_set.count(dest)) {
                            local_edges.push_back({local_src, static_cast<DestID_>(g2l[dest])});
                        }
                    }
                }
                
                if (local_edges.empty()) {
                    ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                        g, nodes, node_set, finalAlgo, useOutdeg,
                        new_ids, current_id);
                    continue;
                }
                
                // Build subgraph CSR
                CSRGraph<NodeID_, DestID_, invert> sub_g =
                    MakeLocalGraphFromELStandalone<NodeID_, DestID_, invert>(local_edges, false);
                
                const size_t sub_N = static_cast<size_t>(sub_g.num_nodes());
                
                // Collect isolated nodes (local IDs not covered by sub_g)
                std::vector<NodeID_> isolated_locals;
                if (sub_N < nodes.size()) {
                    for (size_t i = sub_N; i < nodes.size(); ++i) {
                        isolated_locals.push_back(static_cast<NodeID_>(i));
                    }
                }
                
                // Run Leiden on the subgraph for sub-community detection
                graphbrew::GraphBrewConfig sub_config;
                sub_config.resolution = config.resolution;
                sub_config.maxPasses = config.maxPasses;
                sub_config.maxIterations = config.maxIterations;
                sub_config.aggregation = config.aggregation;
                sub_config.mComputation = config.mComputation;
                sub_config.refinementDepth = config.refinementDepth;
                sub_config.ordering = graphbrew::OrderingStrategy::LAYER;
                
                auto sub_result = graphbrew::runGraphBrew<K>(sub_g, sub_config);
                
                // Classify sub-communities (only iterate over valid membership range)
                std::unordered_map<K, std::vector<NodeID_>> sub_comm_nodes;
                for (NodeID_ lv = 0; lv < static_cast<NodeID_>(sub_N); ++lv) {
                    sub_comm_nodes[sub_result.membership[lv]].push_back(lv);
                }
                
                // Short-circuit: if GraphBrew found ≤1 sub-community, no benefit from splitting
                if (sub_comm_nodes.size() <= 1) {
                    ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                        g, nodes, node_set, finalAlgo, useOutdeg,
                        new_ids, current_id);
                    continue;
                }
                
                // Compute dynamic threshold for sub-communities
                size_t sub_non_empty = sub_comm_nodes.size();
                size_t sub_avg = (sub_non_empty > 0) ? nodes.size() / sub_non_empty : nodes.size();
                size_t sub_threshold = ComputeDynamicMinCommunitySize(
                    nodes.size(), sub_non_empty, sub_avg);
                
                // Separate into large and small sub-communities
                std::vector<K> large_sc_ids, small_sc_ids;
                for (auto& [sc_id, sc_nodes] : sub_comm_nodes) {
                    if (sc_nodes.size() >= sub_threshold) {
                        large_sc_ids.push_back(sc_id);
                    } else {
                        small_sc_ids.push_back(sc_id);
                    }
                }
                
                // ===== SINGLE-PASS EDGE ANALYSIS ON SUB_G =====
                // Simultaneously builds inter-sub-community adjacency (for BFS ordering)
                // and counts internal/cross edges (for cache statistics).
                // Uses sub_g instead of global graph → eliminates g2l hash lookups.
                Timer edgeAnalysisTimer;
                edgeAnalysisTimer.Start();
                std::unordered_map<K, std::unordered_map<K, size_t>> sc_adj;
                size_t comm_internal_edges = 0, comm_cross_edges = 0;
                for (NodeID_ lv = 0; lv < static_cast<NodeID_>(sub_N); ++lv) {
                    K my_sc = sub_result.membership[lv];
                    for (DestID_ neighbor : sub_g.out_neigh(lv)) {
                        NodeID_ nb = static_cast<NodeID_>(neighbor);
                        if (static_cast<size_t>(nb) >= sub_N) continue;
                        K nb_sc = sub_result.membership[nb];
                        if (my_sc == nb_sc) {
                            comm_internal_edges++;
                        } else {
                            comm_cross_edges++;
                            if (sub_comm_nodes[my_sc].size() >= sub_threshold &&
                                sub_comm_nodes[nb_sc].size() >= sub_threshold) {
                                sc_adj[my_sc][nb_sc]++;
                            }
                        }
                    }
                }
                total_internal_edges += comm_internal_edges;
                total_cross_edges += comm_cross_edges;
                edgeAnalysisTimer.Stop();
                time_edge_analysis += edgeAnalysisTimer.Seconds();
                
                // ===== CACHE-AWARE INTER-SUB-COMMUNITY ORDERING =====
                // BFS on pre-computed adjacency: densely connected sub-communities
                // get adjacent placement for better cache reuse.
                std::vector<K> ordered_large_sc;
                if (large_sc_ids.size() > 2) {
                    K start_sc = large_sc_ids[0];
                    size_t max_sc_size = 0;
                    for (K sc_id : large_sc_ids) {
                        if (sub_comm_nodes[sc_id].size() > max_sc_size) {
                            max_sc_size = sub_comm_nodes[sc_id].size();
                            start_sc = sc_id;
                        }
                    }
                    
                    std::unordered_set<K> visited;
                    std::queue<K> bfs_q;
                    bfs_q.push(start_sc);
                    visited.insert(start_sc);
                    
                    while (!bfs_q.empty()) {
                        K cur = bfs_q.front();
                        bfs_q.pop();
                        ordered_large_sc.push_back(cur);
                        
                        std::vector<std::pair<size_t, K>> neighbors;
                        if (sc_adj.count(cur)) {
                            for (auto& [nb, cnt] : sc_adj[cur]) {
                                if (!visited.count(nb)) neighbors.push_back({cnt, nb});
                            }
                        }
                        std::sort(neighbors.begin(), neighbors.end(), std::greater<>());
                        for (auto& [cnt, nb] : neighbors) {
                            if (visited.insert(nb).second) {
                                bfs_q.push(nb);
                            }
                        }
                    }
                    for (K sc_id : large_sc_ids) {
                        if (!visited.count(sc_id)) ordered_large_sc.push_back(sc_id);
                    }
                } else {
                    ordered_large_sc = large_sc_ids;
                }
                
                // Cache-fit statistics (size-based, no edge scanning needed)
                size_t sc_fits_l2 = 0, sc_fits_llc = 0;
                for (K sc_id : ordered_large_sc) {
                    size_t sc_sz = sub_comm_nodes[sc_id].size();
                    total_sub_comms++;
                    if (sc_sz <= l2_node_capacity) { sc_fits_l2++; fits_l2++; }
                    else if (sc_sz <= llc_node_capacity) { sc_fits_llc++; fits_llc++; }
                }
                for (K sc_id : small_sc_ids) total_sub_comms++;
                
                // Allocate reusable mapping vector for sub-community reordering
                std::vector<NodeID_> sc_map(sub_N, static_cast<NodeID_>(-1));
                
                double comm_locality = (comm_internal_edges + comm_cross_edges > 0) ?
                    100.0 * comm_internal_edges / (comm_internal_edges + comm_cross_edges) : 100.0;
                printf("  Community %u (%zu nodes): %zu sub-comms (%zu large, %zu small",
                       comm_id, nodes.size(), sub_non_empty,
                       ordered_large_sc.size(), small_sc_ids.size());
                if (!isolated_locals.empty()) {
                    printf(", %zu isolated", isolated_locals.size());
                }
                printf("), cache-fit: %zu/L2 %zu/LLC, locality: %.0f%%\n",
                       sc_fits_l2, sc_fits_llc, comm_locality);
                
                // ===== Process large sub-communities in cache-aware order =====
                Timer subReorderTimer;
                subReorderTimer.Start();
                for (K sc_id : ordered_large_sc) {
                    auto& sc_local_nodes = sub_comm_nodes[sc_id];
                    size_t sc_size = sc_local_nodes.size();
                    
                    // Select algorithm: adaptive or fixed
                    ReorderingAlgo sc_algo = selectSubAlgo(sc_local_nodes, sub_g, sc_size);
                    algo_distribution[static_cast<int>(sc_algo)]++;
                    reorderFromLocalSubgraph(sc_local_nodes, sub_g, l2g, sc_algo, sc_map);
                }
                
                // Collect small sub-community + isolated nodes
                std::vector<NodeID_> sub_small_nodes;
                for (K sc_id : small_sc_ids) {
                    for (NodeID_ lv : sub_comm_nodes[sc_id]) {
                        sub_small_nodes.push_back(lv);
                    }
                }
                for (NodeID_ lv : isolated_locals) {
                    sub_small_nodes.push_back(lv);
                }
                
                // Handle merged small sub-community + isolated nodes
                if (!sub_small_nodes.empty()) {
                    if (sub_small_nodes.size() >= 100) {
                        ReorderingAlgo small_algo = selectSubAlgo(sub_small_nodes, sub_g, sub_small_nodes.size());
                        reorderFromLocalSubgraph(sub_small_nodes, sub_g, l2g, small_algo, sc_map);
                    } else {
                        std::vector<std::pair<int64_t, NodeID_>> deg_nodes;
                        for (NodeID_ lv : sub_small_nodes) {
                            NodeID_ gv = (static_cast<size_t>(lv) < l2g.size()) ? l2g[lv] : lv;
                            deg_nodes.push_back({-(useOutdeg ? g.out_degree(gv) : g.in_degree(gv)), gv});
                        }
                        std::sort(deg_nodes.begin(), deg_nodes.end());
                        for (auto& [nd, gv] : deg_nodes) {
                            new_ids[gv] = current_id++;
                        }
                    }
                }
                subReorderTimer.Stop();
                time_sub_reorder += subReorderTimer.Seconds();
            }
            
            // ===== Print cache-locality summary =====
            if (total_sub_comms > 0) {
                double internal_ratio = (total_internal_edges + total_cross_edges > 0) ?
                    100.0 * total_internal_edges / (total_internal_edges + total_cross_edges) : 0.0;
                printf("GraphBrew cache analysis:\n");
                printf("  L2=%zuKB (%zu nodes), LLC=%zuMB (%zu nodes), ~%zuB/node\n",
                       l2_bytes / 1024, l2_node_capacity,
                       llc_bytes / (1024*1024), llc_node_capacity, bytes_per_node);
                printf("  sub-communities: %zu total, %zu fit L2, %zu fit LLC, %zu exceed LLC\n",
                       total_sub_comms, fits_l2, fits_llc,
                       total_sub_comms - fits_l2 - fits_llc);
                printf("  edge locality: %.1f%% internal, cross-edges=%zu\n",
                       internal_ratio, total_cross_edges / 2);
                printf("  phase timing: edge-analysis=%.4fs, sub-reorder=%.4fs\n",
                       time_edge_analysis, time_sub_reorder);
                if (autoSubAlgo) {
                    printf("  sub-algo distribution:");
                    for (auto& [algo_id, count] : algo_distribution) {
                        printf(" %s=%zu",
                               ReorderingAlgoStr(static_cast<ReorderingAlgo>(algo_id)).c_str(), count);
                    }
                    printf("\n");
                } else {
                    printf("  sub-algo: fixed %s\n",
                           ReorderingAlgoStr(fixedSubAlgo).c_str());
                }
                
                // Spatial locality estimate: sample edge ID distances
                size_t locality_samples = 0;
                double avg_distance = 0;
                for (int64_t v = 0; v < std::min(N, int64_t(50000)); ++v) {
                    if (new_ids[v] == static_cast<NodeID_>(-1)) continue;
                    for (DestID_ neighbor : g.out_neigh(v)) {
                        NodeID_ u = static_cast<NodeID_>(neighbor);
                        if (new_ids[u] == static_cast<NodeID_>(-1)) continue;
                        avg_distance += std::abs(static_cast<int64_t>(new_ids[v]) - static_cast<int64_t>(new_ids[u]));
                        locality_samples++;
                    }
                }
                if (locality_samples > 0) {
                    avg_distance /= locality_samples;
                    double random_baseline = static_cast<double>(N) / 3.0;
                    printf("  spatial locality: avg edge distance=%.0f (random baseline=%.0f, ratio=%.2fx)\n",
                           avg_distance, random_baseline, avg_distance / random_baseline);
                }
            }
        } else {
            // ===== FLAT MODE (depth=0): apply final algo directly per community =====
            for (auto& [size, comm_id] : sorted_large_comms) {
                auto& nodes = comm_nodes[comm_id];
                if (nodes.empty()) continue;
                
                std::unordered_set<NodeID_> node_set(nodes.begin(), nodes.end());
                
                ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                    g, nodes, node_set, finalAlgo, useOutdeg,
                    new_ids, current_id);
            }
        }
        
        perCommTimer.Stop();
        printf("GraphBrew: per-community reorder: %.4fs\n", perCommTimer.Seconds());
        
        // Verify all nodes assigned
        for (int64_t v = 0; v < N; ++v) {
            if (new_ids[v] == static_cast<NodeID_>(-1)) {
                new_ids[v] = current_id++;
            }
        }
        
        totalTimer.Stop();
        PrintTime("GraphBrew Total Time", totalTimer.Seconds());
    }
    
};

#endif // BUILDER_H_
