// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <parallel/algorithm>
#include <set>
#include <type_traits>
#include <utility>

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "sliding_queue.h"
#include "timer.h"
#include "util.h"

/*
   GAP Benchmark Suite
   Class:  BuilderBase
   Author: Scott Beamer

   Given arguments from the command line (cli), returns a built graph
   - MakeGraph() will parse cli and obtain edgelist to call
   MakeGraphFromEL(edgelist) to perform the actual graph construction
   - edgelist can be from file (Reader) or synthetically generated (Generator)
   - Common case: BuilderBase typedef'd (w/ params) to be Builder (benchmark.h)
 */

//
// A demo program of reordering using Rabbit Order.
#include "edge_list.hpp"
#include "rabbit_order.hpp"
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>

using namespace edge_list;
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

/*
   MIT License

   Copyright (c) 2016, Hao Wei.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
 */
// #include "GoGraph.h"
// #include "GoUtil.h"

template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_, bool invert = true>
class BuilderBase {
typedef EdgePair<NodeID_, DestID_> Edge;
typedef pvector<Edge> EdgeList;

const CLBase &cli_;
bool symmetrize_;
bool needs_weights_;
bool in_place_ = false;
int64_t num_nodes_ = -1;
std::vector<std::pair<ReorderingAlgo, std::string> > reorder_options_;

public:
explicit BuilderBase(const CLBase &cli) : cli_(cli) {

  symmetrize_ = cli_.symmetrize();
  needs_weights_ = !std::is_same<NodeID_, DestID_>::value;
  in_place_ = cli_.in_place();
  // reorder_options_(cli_.reorder_options());
  if (in_place_ && needs_weights_) {
    std::cout << "In-place building (-m) does not support weighted graphs"
              << std::endl;
    exit(-30);
  }
}

DestID_ GetSource(EdgePair<NodeID_, NodeID_> e) {
  return e.u;
}

DestID_ GetSource(EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_> > e) {
  return NodeWeight<NodeID_, WeightT_>(e.u, e.v.w);
}

NodeID_ FindMaxNodeID(const EdgeList &el) {
  NodeID_ max_seen = 0;
#pragma omp parallel for reduction(max : max_seen)
  for (auto it = el.begin(); it < el.end(); it++) {
    Edge e = *it;
    max_seen = __gnu_parallel::max(max_seen, e.u);
    max_seen = __gnu_parallel::max(max_seen, (NodeID_)e.v);
  }
  return max_seen;
}

pvector<NodeID_> CountDegrees(const EdgeList &el, bool transpose) {
  pvector<NodeID_> degrees(num_nodes_, 0);
#pragma omp parallel for
  for (auto it = el.begin(); it < el.end(); it++) {
    Edge e = *it;
    if (symmetrize_ || (!symmetrize_ && !transpose))
      fetch_and_add(degrees[e.u], 1);
    if ((symmetrize_ && !in_place_) || (!symmetrize_ && transpose))
      fetch_and_add(degrees[(NodeID_)e.v], 1);
  }
  return degrees;
}

static pvector<SGOffset> PrefixSum(const pvector<NodeID_> &degrees) {
  pvector<SGOffset> sums(degrees.size() + 1);
  SGOffset total = 0;
  for (size_t n = 0; n < degrees.size(); n++) {
    sums[n] = total;
    total += degrees[n];
  }
  sums[degrees.size()] = total;
  return sums;
}

static pvector<SGOffset> ParallelPrefixSum(const pvector<NodeID_> &degrees) {
  const size_t block_size = 1 << 20;
  const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
  pvector<SGOffset> local_sums(num_blocks);
#pragma omp parallel for
  for (size_t block = 0; block < num_blocks; block++) {
    SGOffset lsum = 0;
    size_t block_end = std::min((block + 1) * block_size, degrees.size());
    for (size_t i = block * block_size; i < block_end; i++)
      lsum += degrees[i];
    local_sums[block] = lsum;
  }
  pvector<SGOffset> bulk_prefix(num_blocks + 1);
  SGOffset total = 0;
  for (size_t block = 0; block < num_blocks; block++) {
    bulk_prefix[block] = total;
    total += local_sums[block];
  }
  bulk_prefix[num_blocks] = total;
  pvector<SGOffset> prefix(degrees.size() + 1);
#pragma omp parallel for
  for (size_t block = 0; block < num_blocks; block++) {
    SGOffset local_total = bulk_prefix[block];
    size_t block_end = std::min((block + 1) * block_size, degrees.size());
    for (size_t i = block * block_size; i < block_end; i++) {
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
               DestID_ ***sq_index, DestID_ **sq_neighs) {
  pvector<NodeID_> diffs(g.num_nodes());
  DestID_ *n_start, *n_end;
#pragma omp parallel for private(n_start, n_end)
  for (NodeID_ n = 0; n < g.num_nodes(); n++) {
    if (transpose) {
      n_start = g.in_neigh(n).begin();
      n_end = g.in_neigh(n).end();
    } else {
      n_start = g.out_neigh(n).begin();
      n_end = g.out_neigh(n).end();
    }
    __gnu_parallel::stable_sort(n_start, n_end);
    DestID_ *new_end = std::unique(n_start, n_end);
    new_end = std::remove(n_start, new_end, n);
    diffs[n] = new_end - n_start;
  }
  pvector<SGOffset> sq_offsets = ParallelPrefixSum(diffs);
  *sq_neighs = new DestID_[sq_offsets[g.num_nodes()]];
  *sq_index = CSRGraph<NodeID_, DestID_>::GenIndex(sq_offsets, *sq_neighs);
#pragma omp parallel for private(n_start)
  for (NodeID_ n = 0; n < g.num_nodes(); n++) {
    if (transpose)
      n_start = g.in_neigh(n).begin();
    else
      n_start = g.out_neigh(n).begin();
    std::copy(n_start, n_start + diffs[n], (*sq_index)[n]);
  }
}

CSRGraph<NodeID_, DestID_, invert>
SquishGraph(const CSRGraph<NodeID_, DestID_, invert> &g) {
  DestID_ **out_index, *out_neighs, **in_index, *in_neighs;
  SquishCSR(g, false, &out_index, &out_neighs);
  if (g.directed()) {
    if (invert)
      SquishCSR(g, true, &in_index, &in_neighs);
    return CSRGraph<NodeID_, DestID_, invert>(
      g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  } else {
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
                    DestID_ ***inv_index, DestID_ **inv_neighs) {
  // preprocess EdgeList - sort & squish in place
  __gnu_parallel::stable_sort(el.begin(), el.end());
  auto new_end = std::unique(el.begin(), el.end());
  el.resize(new_end - el.begin());
  auto self_loop = [](Edge e) {
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
  if (!symmetrize_) { // not going to symmetrize so no need to add edges
    size_t new_size = num_edges * sizeof(DestID_);
    *neighs = static_cast<DestID_ *>(std::realloc(*neighs, new_size));
    *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
    if (invert) { // create inv_neighs & inv_index for incoming edges
      pvector<SGOffset> inoffsets = ParallelPrefixSum(indegrees);
      *inv_neighs = new DestID_[inoffsets[num_nodes_]];
      *inv_index =
        CSRGraph<NodeID_, DestID_>::GenIndex(inoffsets, *inv_neighs);
      for (NodeID_ u = 0; u < num_nodes_; u++) {
        for (DestID_ *it = (*index)[u]; it < (*index)[u + 1]; it++) {
          NodeID_ v = static_cast<NodeID_>(*it);
          (*inv_neighs)[inoffsets[v]] = u;
          inoffsets[v]++;
        }
      }
    }
  } else { // symmetrize graph by adding missing inverse edges
    // Step 1 - count number of needed inverses
    pvector<NodeID_> invs_needed(num_nodes_, 0);
    for (NodeID_ u = 0; u < num_nodes_; u++) {
      for (SGOffset i = offsets[u]; i < offsets[u + 1]; i++) {
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
    for (NodeID_ n = 0; n < num_nodes_; n++) {
      offsets[n] += total_missing_inv;
      total_missing_inv += invs_needed[n];
    }
    offsets[num_nodes_] += total_missing_inv;
    size_t newsize = (offsets[num_nodes_] * sizeof(DestID_));
    *neighs = static_cast<DestID_ *>(std::realloc(*neighs, newsize));
    if (*neighs == nullptr) {
      std::cout << "Call to realloc() failed" << std::endl;
      exit(-33);
    }
    // Step 2 - spread out existing neighs to make room for inverses
    //   copies backwards (overwrites) and inserts free space at starts
    SGOffset tail_index = offsets[num_nodes_] - 1;
    for (NodeID_ n = num_nodes_ - 1; n >= 0; n--) {
      SGOffset new_start = offsets[n] + invs_needed[n];
      for (SGOffset i = offsets[n + 1] - 1; i >= new_start; i--) {
        (*neighs)[tail_index] = (*neighs)[i - total_missing_inv];
        tail_index--;
      }
      total_missing_inv -= invs_needed[n];
      tail_index -= invs_needed[n];
    }
    // Step 3 - add missing inverse edges into free spaces from Step 2
    for (NodeID_ u = 0; u < num_nodes_; u++) {
      for (SGOffset i = offsets[u] + invs_needed[u]; i < offsets[u + 1];
           i++) {
        DestID_ v = (*neighs)[i];
        bool inv_found = std::binary_search(
          *neighs + offsets[v] + invs_needed[v], *neighs + offsets[v + 1],
          static_cast<DestID_>(u));
        if (!inv_found) {
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
             DestID_ **neighs) {
  pvector<NodeID_> degrees = CountDegrees(el, transpose);
  pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
  *neighs = new DestID_[offsets[num_nodes_]];
  *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
#pragma omp parallel for
  for (auto it = el.begin(); it < el.end(); it++) {
    Edge e = *it;
    if (symmetrize_ || (!symmetrize_ && !transpose))
      (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
    if (symmetrize_ || (!symmetrize_ && transpose))
      (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
        GetSource(e);
  }
}

CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList &el) {
  DestID_ **index = nullptr, **inv_index = nullptr;
  DestID_ *neighs = nullptr, *inv_neighs = nullptr;
  Timer t;
  t.Start();
  if (num_nodes_ == -1)
    num_nodes_ = FindMaxNodeID(el) + 1;
  if (needs_weights_)
    Generator<NodeID_, DestID_, WeightT_>::InsertWeights(el);
  if (in_place_) {
    MakeCSRInPlace(el, &index, &neighs, &inv_index, &inv_neighs);
  } else {
    MakeCSR(el, false, &index, &neighs);
    if (!symmetrize_ && invert) {
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

CSRGraph<NodeID_, DestID_, invert> MakeGraph() {
  CSRGraph<NodeID_, DestID_, invert> g;
  CSRGraph<NodeID_, DestID_, invert> g_final;
  { // extra scope to trigger earlier deletion of el (save memory)
    EdgeList el;
    if (cli_.filename() != "") {
      Reader<NodeID_, DestID_, WeightT_, invert> r(cli_.filename());
      if ((r.GetSuffix() == ".sg") || (r.GetSuffix() == ".wsg")) {
        return r.ReadSerializedGraph();
      } else {
        el = r.ReadFile(needs_weights_);
      }
    } else if (cli_.scale() != -1) {
      Generator<NodeID_, DestID_> gen(cli_.scale(), cli_.degree());
      el = gen.GenerateEL(cli_.uniform());
    }
    g = MakeGraphFromEL(el);
  }

  if (in_place_)
    g_final = g;
  else
    g_final = SquishGraph(g);

  pvector<NodeID_> new_ids(g.num_nodes());
  for (const auto &option : cli_.reorder_options()) {
    new_ids.fill(UINT_E_MAX);
    if (!option.second.empty()) {
      GenerateMapping(g_final, new_ids, option.first, true, option.second);
    } else {
      GenerateMapping(g_final, new_ids, option.first, true);
    }
    g_final = RelabelByMapping(g_final, new_ids);
  }
  return g_final;
}

// Relabels (and rebuilds) graph by order of decreasing degree
static CSRGraph<NodeID_, DestID_, invert>
RelabelByDegree(const CSRGraph<NodeID_, DestID_, invert> &g) {
  if (g.directed()) {
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
  for (NodeID_ n = 0; n < g.num_nodes(); n++) {
    degrees[n] = degree_id_pairs[n].first;
    new_ids[degree_id_pairs[n].second] = n;
  }
  pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
  DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
  DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for
  for (NodeID_ u = 0; u < g.num_nodes(); u++) {
    for (NodeID_ v : g.out_neigh(u))
      neighs[offsets[new_ids[u]]++] = new_ids[v];
    __gnu_parallel::stable_sort(index[new_ids[u]], index[new_ids[u] + 1]);
  }
  t.Stop();
  PrintTime("Relabel", t.Seconds());
  return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
}

static CSRGraph<NodeID_, DestID_, invert>
RelabelByMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                 pvector<NodeID_> &new_ids) {
  Timer t;
  DestID_ **out_index;
  DestID_ *out_neighs;
  DestID_ **in_index;
  DestID_ *in_neighs;
  CSRGraph<NodeID_, DestID_, invert> g_relabel;

  t.Start();
  pvector<NodeID_> out_degrees(g.num_nodes(), 0);

#pragma omp parallel for
  for (NodeID_ n = 0; n < g.num_nodes(); n++) {
    out_degrees[new_ids[n]] = g.out_degree(n);
  }
  pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
  out_neighs = new DestID_[out_offsets[g.num_nodes()]];
  out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
#pragma omp parallel for
  for (NodeID_ u = 0; u < g.num_nodes(); u++) {
    for (NodeID_ v : g.out_neigh(u))
      out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
    __gnu_parallel::stable_sort(out_index[new_ids[u]],
                                out_index[new_ids[u] + 1]);
  }

  if (g.directed()) {
    pvector<NodeID_> in_degrees(g.num_nodes(), 0);
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      in_degrees[new_ids[n]] = g.in_degree(n);
    }
    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      __gnu_parallel::stable_sort(in_index[new_ids[u]],
                                  in_index[new_ids[u] + 1]);
    }
    t.Stop();
    g_relabel = CSRGraph<NodeID_, DestID_, invert>(
      g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  } else {
    t.Stop();
    g_relabel = CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                                                   out_neighs);
  }
  PrintTime("Relabel", t.Seconds());
  return g_relabel;
}

const string ReorderingAlgoStr(ReorderingAlgo type) {
  switch (type) {
  case HubSort:
    return "HubSort";
  case DBG:
    return "DBG";
  case HubClusterDBG:
    return "HubClusterDBG";
  case HubSortDBG:
    return "HubSortDBG";
  case HubCluster:
    return "HubCluster";
  case Random:
    return "Random";
  case RabbitOrder:
    return "RabbitOrder";
  case GOrder:
    return "GOrder";
  case ORIGINAL:
    return "Original";
  case Sort:
    return "Sort";
  case MAP:
    return "MAP";
  default:
    std::cout << "Unknown PreprocessTypeStr type: " << type << std::endl;
    abort();
  }
  assert(0);
  return "";
}

void GenerateMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                     pvector<NodeID_> &new_ids,
                     ReorderingAlgo reordering_algo, bool useOutdeg,
                     std::string map_file = "mapping.label") {
  switch (reordering_algo) {
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
    GenerateRandomMapping(g, new_ids, useOutdeg);
    break;
  case RabbitOrder:
    GenerateRabbitOrderMapping(g, new_ids);
    break;
  case GOrder:
    // GenerateGOrderMapping(g, new_ids);
    break;
  case MAP:
    LoadMappingFromFile(g, new_ids, useOutdeg, map_file);
    break;
  case ORIGINAL:
    std::cout << "Should not be here!" << std::endl;
    std::abort();
    return;
    break;
  default:
    std::cout << "Unknown generateMapping type: " << reordering_algo
              << std::endl;
    std::abort();
  }
#ifdef _DEBUG
  VerifyMapping(g, new_ids);
  // exit(-1);
#endif
}

void VerifyMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                   const pvector<NodeID_> &new_ids) {
  NodeID_ *hist = alloc_align_4k<NodeID_>(g.num_nodes());
  int64_t num_nodes = g.num_nodes();

#pragma omp parallel for
  for (long i = 0; i < num_nodes; i++) {
    hist[i] = new_ids[i];
  }

  __gnu_parallel::stable_sort(&hist[0], &hist[num_nodes]);

  NodeID_ count = 0;

#pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i++) {
    if (hist[i] != i) {
      __sync_fetch_and_add(&count, 1);
    }
  }

  if (count != 0) {
    std::cout << "Num of vertices did not match: " << count << std::endl;
    std::cout << "Mapping is invalid.!" << std::endl;
    std::abort();
  } else {
    std::cout << "Mapping is valid.!" << std::endl;
  }
  std::free(hist);
}

void LoadMappingFromFile(const CSRGraph<NodeID_, DestID_, invert> &g,
                         pvector<NodeID_> &new_ids, bool useOutdeg,
                         std::string map_file = "mapping.label") {
  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();

  ifstream ifs(map_file.c_str(), std::ifstream::in);
  if (!ifs.good()) {
    cout << "File " << map_file << " does not exist!" << endl;
    exit(-1);
  }
  int64_t num_nodes_1, num_edges_1;
  ifs >> num_nodes_1;
  ifs >> num_edges_1;
  cout << " num_nodes: " << num_nodes_1 << " num_edges: " << num_edges_1
       << endl;
  cout << " num_nodes: " << num_nodes << " num_edges: " << num_edges << endl;
  if (num_nodes != num_nodes_1) {
    cout << "Mismatch: " << num_nodes << " " << num_nodes_1 << endl;
    exit(-1);
  }
  if (num_nodes != (int64_t)new_ids.size()) {
    cout << "Mismatch: " << num_nodes << " " << new_ids.size() << endl;
    exit(-1);
  }
  if (num_edges != num_edges_1) {
    cout << "Warning! Potential mismatch: " << num_edges << " " << num_edges_1
         << endl;
  }
  char c;
  unsigned long int st, v;
  bool tab = true;
  if (tab) {
    for (int64_t i = 0; i < num_nodes; i++) {
      ifs >> st >> v;
      new_ids[st] = v;
    }
  } else {
    for (int64_t i = 0; i < num_nodes; i++) {
      ifs >> c >> st >> c >> v >> c;
      new_ids[st] = v;
    }
  }
  ifs.close();

  t.Stop();
  PrintTime("Load Map Time", t.Seconds());
}

void GenerateRandomMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                           pvector<NodeID_> &new_ids, bool useOutdeg) {
  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  // int64_t num_edges = g.num_edges();

  NodeID_ granularity = 1;
  NodeID_ slice = (num_nodes - granularity + 1) / granularity;
  NodeID_ artificial_num_nodes = slice * granularity;
  assert(artificial_num_nodes <= num_nodes);
  pvector<NodeID_> slice_index;
  slice_index.resize(slice);

#pragma omp parallel for
  for (NodeID_ i = 0; i < slice; i++) {
    slice_index[i] = i;
  }

  __gnu_parallel::random_shuffle(slice_index.begin(), slice_index.end());

  {
#pragma omp parallel for
    for (NodeID_ i = 0; i < slice; i++) {
      NodeID_ new_index = slice_index[i] * granularity;
      for (NodeID_ j = 0; j < granularity; j++) {
        NodeID_ v = (i * granularity) + j;
        if (v < artificial_num_nodes) {
          new_ids[v] = new_index + j;
        }
      }
    }
  }
  for (NodeID_ i = artificial_num_nodes; i < num_nodes; i++) {
    new_ids[i] = i;
  }
  slice_index.clear();

  t.Stop();
  PrintTime("Random Map Time", t.Seconds());
}

void GenerateHubSortDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids, bool useOutdeg) {

  typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();

  int64_t avgDegree = num_edges / num_nodes;
  size_t hubCount{0};

  const int num_threads = omp_get_max_threads();
  pvector<degree_nodeid_t> local_degree_id_pairs[num_threads];
  int64_t slice = num_nodes / num_threads;
  int64_t start[num_threads];
  int64_t end[num_threads];
  int64_t hub_count[num_threads];
  int64_t non_hub_count[num_threads];
  int64_t new_index[num_threads];
  for (int t = 0; t < num_threads; t++) {
    start[t] = t * slice;
    end[t] = (t + 1) * slice;
    hub_count[t] = 0;
  }
  end[num_threads - 1] = num_nodes;

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int64_t t = 0; t < num_threads; t++) {
    for (int64_t v = start[t]; v < end[t]; ++v) {
      if (useOutdeg) {
        int64_t out_degree_v = g.out_degree(v);
        if (out_degree_v > avgDegree) {
          local_degree_id_pairs[t].push_back(std::make_pair(out_degree_v, v));
        }
      } else {
        int64_t in_degree_v = g.in_degree(v);
        if (in_degree_v > avgDegree) {
          local_degree_id_pairs[t].push_back(std::make_pair(in_degree_v, v));
        }
      }
    }
  }
  for (int t = 0; t < num_threads; t++) {
    hub_count[t] = local_degree_id_pairs[t].size();
    hubCount += hub_count[t];
    non_hub_count[t] = end[t] - start[t] - hub_count[t];
  }
  new_index[0] = hubCount;
  for (int t = 1; t < num_threads; t++) {
    new_index[t] = new_index[t - 1] + non_hub_count[t - 1];
  }
  pvector<degree_nodeid_t> degree_id_pairs(hubCount);

  size_t k = 0;
  for (int i = 0; i < num_threads; i++) {
    for (size_t j = 0; j < local_degree_id_pairs[i].size(); j++) {
      degree_id_pairs[k++] = local_degree_id_pairs[i][j];
    }
    local_degree_id_pairs[i].clear();
  }
  assert(degree_id_pairs.size() == hubCount);
  assert(k == hubCount);

  __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                              std::greater<degree_nodeid_t>());

#pragma omp parallel for
  for (size_t n = 0; n < hubCount; ++n) {
    new_ids[degree_id_pairs[n].second] = n;
  }
  pvector<degree_nodeid_t>().swap(degree_id_pairs);

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int t = 0; t < num_threads; t++) {
    for (int64_t v = start[t]; v < end[t]; ++v) {
      if (new_ids[v] == (NodeID_)UINT_E_MAX) {
        new_ids[v] = new_index[t]++;
      }
    }
  }

  t.Stop();
  PrintTime("HubSortDBG Map Time", t.Seconds());
}

void GenerateHubClusterDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids, bool useOutdeg) {
  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();

  uint32_t avg_vertex = num_edges / num_nodes;

  const int num_buckets = 2;
  avg_vertex = avg_vertex;
  uint32_t bucket_threshold[] = {avg_vertex, static_cast<uint32_t>(-1)};

  vector<uint32_t> bucket_vertices[num_buckets];
  const int num_threads = omp_get_max_threads();
  vector<uint32_t> local_buckets[num_threads][num_buckets];

  if (useOutdeg) {
    // This loop relies on a static scheduling
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_nodes; i++) {
      for (unsigned int j = 0; j < num_buckets; j++) {
        const int64_t &count = g.out_degree(i);
        if (count <= bucket_threshold[j]) {
          local_buckets[omp_get_thread_num()][j].push_back(i);
          break;
        }
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_nodes; i++) {
      for (unsigned int j = 0; j < num_buckets; j++) {
        const int64_t &count = g.in_degree(i);
        if (count <= bucket_threshold[j]) {
          local_buckets[omp_get_thread_num()][j].push_back(i);
          break;
        }
      }
    }
  }

  int temp_k = 0;
  uint32_t start_k[num_threads][num_buckets];
  for (int32_t j = num_buckets - 1; j >= 0; j--) {
    for (int t = 0; t < num_threads; t++) {
      start_k[t][j] = temp_k;
      temp_k += local_buckets[t][j].size();
    }
  }

#pragma omp parallel for schedule(static)
  for (int t = 0; t < num_threads; t++) {
    for (int32_t j = num_buckets - 1; j >= 0; j--) {
      const vector<uint32_t> &current_bucket = local_buckets[t][j];
      int k = start_k[t][j];
      const size_t &size = current_bucket.size();
      for (uint32_t i = 0; i < size; i++) {
        new_ids[current_bucket[i]] = k++;
      }
    }
  }

  for (int i = 0; i < num_threads; i++) {
    for (unsigned int j = 0; j < num_buckets; j++) {
      local_buckets[i][j].clear();
    }
  }

  t.Stop();
  PrintTime("HubClusterDBG Map Time", t.Seconds());
}

void GenerateHubSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                            pvector<NodeID_> &new_ids, bool useOutdeg) {

  typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();

  pvector<degree_nodeid_t> degree_id_pairs(num_nodes);
  int64_t avgDegree = num_edges / num_nodes;
  size_t hubCount{0};

  /* STEP I - collect degrees of all vertices */
#pragma omp parallel for reduction(+ : hubCount)
  for (int64_t v = 0; v < num_nodes; ++v) {
    if (useOutdeg) {
      int64_t out_degree_v = g.out_degree(v);
      degree_id_pairs[v] = std::make_pair(out_degree_v, v);
      if (out_degree_v > avgDegree) {
        ++hubCount;
      }
    } else {
      int64_t in_degree_v = g.in_degree(v);
      degree_id_pairs[v] = std::make_pair(in_degree_v, v);
      if (in_degree_v > avgDegree) {
        ++hubCount;
      }
    }
  }

  /* Step II - sort the degrees in parallel */
  __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                              std::greater<degree_nodeid_t>());

  /* Step III - make a remap based on the sorted degree list [Only for hubs]
   */
#pragma omp parallel for
  for (size_t n = 0; n < hubCount; ++n) {
    new_ids[degree_id_pairs[n].second] = n;
  }
  // clearing space from degree pairs
  pvector<degree_nodeid_t>().swap(degree_id_pairs);

  /* Step IV - assigning a remap for (easy) non hub vertices */
  auto numHubs = hubCount;
  SlidingQueue<int64_t> queue(numHubs);
#pragma omp parallel
  {
    QueueBuffer<int64_t> lqueue(queue, numHubs / omp_get_max_threads());
#pragma omp for
    for (int64_t n = numHubs; n < num_nodes; ++n) {
      if (new_ids[n] == (NodeID_)UINT_E_MAX) {
        // This steps preserves the ordering of the original graph (as much as
        // possible)
        new_ids[n] = n;
      } else {
        int64_t remappedTo = new_ids[n];
        if (new_ids[remappedTo] == (NodeID_)UINT_E_MAX) {
          // safe to swap Ids because the original vertex is a non-hub
          new_ids[remappedTo] = n;
        } else {
          // Cannot swap ids because original vertex was a hub (swapping
          // would disturb sorted ordering of hubs - not allowed)
          lqueue.push_back(n);
        }
      }
    }
    lqueue.flush();
  }
  queue.slide_window(); // the queue keeps a list of vertices where a simple
                        // swap of locations is not possible
  /* Step V - assigning remaps for remaining non hubs */
  int64_t unassignedCtr{0};
  auto q_iter = queue.begin();
#pragma omp parallel for
  for (size_t n = 0; n < numHubs; ++n) {
    if (new_ids[n] == (NodeID_)UINT_E_MAX) {
      int64_t u = *(q_iter + __sync_fetch_and_add(&unassignedCtr, 1));
      new_ids[n] = u;
    }
  }

  t.Stop();
  PrintTime("HubSort Map Time", t.Seconds());
}

void GenerateSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                         pvector<NodeID_> &new_ids, bool useOutdeg) {

  typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  // int64_t num_edges = g.num_edges();

  pvector<degree_nodeid_t> degree_id_pairs(num_nodes);

  if (useOutdeg) {
#pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
      int64_t out_degree_v = g.out_degree(v);
      degree_id_pairs[v] = std::make_pair(out_degree_v, v);
    }
  } else {
#pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
      int64_t in_degree_v = g.in_degree(v);
      degree_id_pairs[v] = std::make_pair(in_degree_v, v);
    }
  }

  __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                              std::greater<degree_nodeid_t>());

#pragma omp parallel for
  for (int64_t n = 0; n < num_nodes; ++n) {
    new_ids[degree_id_pairs[n].second] = n;
  }

  pvector<degree_nodeid_t>().swap(degree_id_pairs);

  t.Stop();
  PrintTime("Sort Map Time", t.Seconds());
}

void GenerateDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                        pvector<NodeID_> &new_ids, bool useOutdeg) {
  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();

  uint32_t avg_vertex = num_edges / num_nodes;
  const uint32_t &av = avg_vertex;

  uint32_t bucket_threshold[] = {
    av / 2,   av,       av * 2,   av * 4,
    av * 8,   av * 16,  av * 32,  av * 64,
    av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
  int num_buckets = 8;
  if (num_buckets > 11) {
    // if you really want to increase the bucket count, add more thresholds to
    // the bucket_threshold above.
    std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
    assert(0);
  }
  bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);

  vector<uint32_t> bucket_vertices[num_buckets];
  const int num_threads = omp_get_max_threads();
  vector<uint32_t> local_buckets[num_threads][num_buckets];

  if (useOutdeg) {
    // This loop relies on a static scheduling
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_buckets; j++) {
        const int64_t &count = g.out_degree(i);
        if (count <= bucket_threshold[j]) {
          local_buckets[omp_get_thread_num()][j].push_back(i);
          break;
        }
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_buckets; j++) {
        const int64_t &count = g.in_degree(i);
        if (count <= bucket_threshold[j]) {
          local_buckets[omp_get_thread_num()][j].push_back(i);
          break;
        }
      }
    }
  }

  int temp_k = 0;
  uint32_t start_k[num_threads][num_buckets];
  for (int32_t j = num_buckets - 1; j >= 0; j--) {
    for (int t = 0; t < num_threads; t++) {
      start_k[t][j] = temp_k;
      temp_k += local_buckets[t][j].size();
    }
  }

#pragma omp parallel for schedule(static)
  for (int t = 0; t < num_threads; t++) {
    for (int j = num_buckets - 1; j >= 0; j--) {
      const vector<uint32_t> &current_bucket = local_buckets[t][j];
      int k = start_k[t][j];
      const size_t &size = current_bucket.size();
      for (uint32_t i = 0; i < size; i++) {
        new_ids[current_bucket[i]] = k++;
      }
    }
  }

  for (int i = 0; i < num_threads; i++) {
    for (int j = 0; j < num_buckets; j++) {
      local_buckets[i][j].clear();
    }
  }

  t.Stop();
  PrintTime("DBG Map Time", t.Seconds());
}

void GenerateHubClusterMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids, bool useOutdeg) {

  typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

  Timer t;
  t.Start();

  int64_t num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges();

  pvector<degree_nodeid_t> degree_id_pairs(num_nodes);
  int64_t avgDegree = num_edges / num_nodes;
  // size_t hubCount {0};

  const int PADDING = 64 / sizeof(uintE);
  int64_t *localOffsets = new int64_t[omp_get_max_threads() * PADDING]();
  int64_t partitionSz = num_nodes / omp_get_max_threads();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int startID = partitionSz * tid;
    int stopID = partitionSz * (tid + 1);
    if (tid == omp_get_max_threads() - 1) {
      stopID = num_nodes;
    }
    for (int n = startID; n < stopID; ++n) {
      if (useOutdeg) {
        int64_t out_degree_n = g.out_degree(n);
        if (out_degree_n > avgDegree) {
          ++localOffsets[tid * PADDING];
          new_ids[n] = 1;
        }
      } else {
        int64_t in_degree_n = g.in_degree(n);
        if (in_degree_n > avgDegree) {
          ++localOffsets[tid * PADDING];
          new_ids[n] = 1;
        }
      }
    }
  }
  int64_t sum{0};
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    auto origCount = localOffsets[tid * PADDING];
    localOffsets[tid * PADDING] = sum;
    sum += origCount;
  }

  /* Step II - assign a remap for the hub vertices first */
#pragma omp parallel
  {
    int64_t localCtr{0};
    int tid = omp_get_thread_num();
    int64_t startID = partitionSz * tid;
    int64_t stopID = partitionSz * (tid + 1);
    if (tid == omp_get_max_threads() - 1) {
      stopID = num_nodes;
    }
    for (int64_t n = startID; n < stopID; ++n) {
      if (new_ids[n] != (NodeID_)UINT_E_MAX) {
        new_ids[n] = (NodeID_)localOffsets[tid * PADDING] + (NodeID_)localCtr;
        ++localCtr;
      }
    }
  }
  delete[] localOffsets;

  /* Step III - assigning a remap for (easy) non hub vertices */
  auto numHubs = sum;
  SlidingQueue<int64_t> queue(numHubs);
#pragma omp parallel
  {
    // assert(omp_get_max_threads() == 56);
    QueueBuffer<int64_t> lqueue(queue, numHubs / omp_get_max_threads());
#pragma omp for
    for (int64_t n = numHubs; n < num_nodes; ++n) {
      if (new_ids[n] == (NodeID_)UINT_E_MAX) {
        // This steps preserves the ordering of the original graph (as much as
        // possible)
        new_ids[n] = (NodeID_)n;
      } else {
        int64_t remappedTo = new_ids[n];
        if (new_ids[remappedTo] == (NodeID_)UINT_E_MAX) {
          // safe to swap Ids because the original vertex is a non-hub
          new_ids[remappedTo] = (NodeID_)n;
        } else {
          // Cannot swap ids because original vertex was a hub (swapping
          // would disturb sorted ordering of hubs - not allowed)
          lqueue.push_back(n);
        }
      }
    }
    lqueue.flush();
  }
  queue.slide_window(); // the queue keeps a list of vertices where a simple
                        // swap of locations is not possible

  /* Step IV - assigning remaps for remaining non hubs */
  int64_t unassignedCtr{0};
  auto q_iter = queue.begin();
#pragma omp parallel for
  for (int64_t n = 0; n < numHubs; ++n) {
    if (new_ids[n] == (NodeID_)UINT_E_MAX) {
      int64_t u = *(q_iter + __sync_fetch_and_add(&unassignedCtr, 1));
      new_ids[n] = (NodeID_)u;
    }
  }

  t.Stop();
  PrintTime("HubCluster Map Time", t.Seconds());
}

// @inproceedings{popt-hpca21,
//   title={P-OPT: Practical Optimal Cache Replacement for Graph Analytics},
//   author={Balaji, Vignesh and Crago, Neal and Jaleel, Aamer and Lucia,
//   Brandon}, booktitle={2021 IEEE International Symposium on
//   High-Performance Computer Architecture (HPCA)}, pages={668--681},
//   year={2021},
//   organization={IEEE}
// }

/*
   CSR-segmenting as proposed in the Cagra paper

   Partitions the graphs and produces a sub-graph within a specified range of
   vertices.

   The following implementation assumes use in pull implementation and that
   only the partitioned CSC is required
 */
static CSRGraph<NodeID_, DestID_, invert>
graphSlicer(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ startID,
            NodeID_ stopID, bool outDegree = false,
            bool modifyBothDestlists = false) {
  /* create a partition of a graph in the range [startID, stopID) */
  Timer t;
  t.Start();

  // NOTE: that pull implementation should specify outDegree == false
  //       and push implementations should use outDegree == true

  if (g.directed() == true) {
    /* Step I : For the requested range [startID, stopID), construct the
     * reduced degree per vertex */
    pvector<NodeID_> degrees(
      g.num_nodes()); // note that stopID is not included in the range
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ n = 0; n < g.num_nodes(); ++n) {
      if (outDegree == true) {
        NodeID_ newDegree(0);
        for (NodeID_ m : g.out_neigh(n)) {
          if (m >= startID && m < stopID) {
            ++newDegree;
          }
        }
        degrees[n] = newDegree;
      } else {
        NodeID_ newDegree(0);
        for (NodeID_ m : g.in_neigh(n)) {
          if (m >= startID && m < stopID) {
            ++newDegree;
          }
        }
        degrees[n] = newDegree;
      }
    }

    /* Step II : Construct a trimmed offset list */
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
    if (outDegree == true) {
#pragma omp parallel for schedule(dynamic, 1024)
      for (NodeID_ u = 0; u < g.num_nodes(); ++u) {
        for (NodeID_ v : g.out_neigh(u)) {
          if (v >= startID && v < stopID) {
            neighs[offsets[u]++] = v;
          }
        }
      }
    } else {
#pragma omp parallel for schedule(dynamic, 1024)
      for (NodeID_ u = 0; u < g.num_nodes(); ++u) {
        for (NodeID_ v : g.in_neigh(u)) {
          if (v >= startID && v < stopID) {
            neighs[offsets[u]++] = v;
          }
        }
      }
    }

    /* Step III : Populate the inv dest lists (for push-pull implementations)
     */
    DestID_ *inv_neighs(nullptr);
    DestID_ **inv_index(nullptr);
    if (modifyBothDestlists == true) {
      // allocate space
      pvector<NodeID_> inv_degrees(g.num_nodes());
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); ++u) {
        if (outDegree == true) {
          inv_degrees[u] = g.in_degree(u);
        } else {
          inv_degrees[u] = g.out_degree(u);
        }
      }
      pvector<SGOffset> inv_offsets = ParallelPrefixSum(inv_degrees);
      inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
      inv_index =
        CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);

// populate the inv dest list
#pragma omp parallel for schedule(dynamic, 1024)
      for (NodeID_ u = 0; u < g.num_nodes(); ++u) {
        if (outDegree == true) {
          for (NodeID_ v : g.in_neigh(u)) {
            inv_neighs[inv_offsets[u]++] = v;
          }
        } else {
          for (NodeID_ v : g.out_neigh(u)) {
            inv_neighs[inv_offsets[u]++] = v;
          }
        }
      }
    }

    /* Step IV : return the appropriate graph */
    if (outDegree == true) {
      t.Stop();
      PrintTime("Slice-time", t.Seconds());
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                                                inv_index, inv_neighs);
    } else {
      t.Stop();
      PrintTime("Slice-time", t.Seconds());
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                                                inv_neighs, index, neighs);
    }
  } else {
    /* Step I : For the requested range [startID, stopID), construct the
     * reduced degree per vertex */
    pvector<NodeID_> degrees(
      g.num_nodes()); // note that stopID is not included in the range
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ n = 0; n < g.num_nodes(); ++n) {
      NodeID_ newDegree(0);
      for (NodeID_ m : g.out_neigh(n)) {
        if (m >= startID && m < stopID) {
          ++newDegree; // if neighbor is in current partition
        }
      }
      degrees[n] = newDegree;
    }

    /* Step II : Construct a trimmed offset list */
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ u = 0; u < g.num_nodes(); ++u) {
      for (NodeID_ v : g.out_neigh(u)) {
        if (v >= startID && v < stopID) {
          neighs[offsets[u]++] = v;
        }
      }
    }

    /* Step III : return the appropriate graph */
    t.Stop();
    PrintTime("Slice-time", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
  }
}

static CSRGraph<NodeID_, DestID_, invert>
quantizeGraph(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ numTiles) {

  NodeID_ tileSz = g.num_nodes() / numTiles;
  if (numTiles > g.num_nodes())
    tileSz = 1;
  else if (g.num_nodes() % numTiles != 0)
    tileSz += 1;

  pvector<NodeID_> degrees(g.num_nodes(), 0);
#pragma omp parallel for
  for (NodeID_ n = 0; n < g.num_nodes(); ++n) {
    std::set<NodeID_> uniqNghs;
    for (NodeID_ ngh : g.out_neigh(n)) {
      uniqNghs.insert(ngh / tileSz);
    }
    degrees[n] = uniqNghs.size();
    assert(degrees[n] <= numTiles);
  }

  pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
  DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
  DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for schedule(dynamic, 1024)
  for (NodeID_ u = 0; u < g.num_nodes(); u++) {
    std::set<NodeID_> uniqNghs;
    for (NodeID_ ngh : g.out_neigh(u)) {
      uniqNghs.insert(ngh / tileSz);
    }

    auto it = uniqNghs.begin();
    for (NodeID_ i = 0; i < static_cast<NodeID_>(uniqNghs.size()); ++i) {
      neighs[offsets[u]++] = *it;
      it++;
    }
  }
  return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
}

// Proper degree sorting
static CSRGraph<NodeID_, DestID_, invert>
DegSort(const CSRGraph<NodeID_, DestID_, invert> &g, bool outDegree,
        pvector<NodeID_> &new_ids, bool createOnlyDegList,
        bool createBothCSRs) {
  Timer t;
  t.Start();

  typedef std::pair<int64_t, NodeID_> degree_node_p;
  pvector<degree_node_p> degree_id_pairs(g.num_nodes());
  if (g.directed() == true) {
/* Step I: Create a list of degrees */
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      if (outDegree == true) {
        degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
      } else {
        degree_id_pairs[n] = std::make_pair(g.in_degree(n), n);
      }
    }

    /* Step II: Sort based on degree order */
    __gnu_parallel::sort(
      degree_id_pairs.begin(), degree_id_pairs.end(),
      std::greater<degree_node_p>()); // TODO:Use parallel sort

/* Step III: assigned remap for the hub vertices */
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      new_ids[degree_id_pairs[n].second] = n;
    }

    /* Step VI: generate degree to build a new graph */
    pvector<NodeID_> degrees(g.num_nodes());
    pvector<NodeID_> inv_degrees(g.num_nodes());
    if (outDegree == true) {
#pragma omp parallel for
      for (NodeID_ n = 0; n < g.num_nodes(); n++) {
        degrees[new_ids[n]] = g.out_degree(n);
        inv_degrees[new_ids[n]] = g.in_degree(n);
      }
    } else {
#pragma omp parallel for
      for (NodeID_ n = 0; n < g.num_nodes(); n++) {
        degrees[new_ids[n]] = g.in_degree(n);
        inv_degrees[new_ids[n]] = g.out_degree(n);
      }
    }

    /* Graph building phase */
    pvector<SGOffset> offsets = ParallelPrefixSum(inv_degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      if (outDegree == true) {
        for (NodeID_ v : g.in_neigh(u))
          neighs[offsets[new_ids[u]]++] = new_ids[v];
      } else {
        for (NodeID_ v : g.out_neigh(u))
          neighs[offsets[new_ids[u]]++] = new_ids[v];
      }
      std::sort(index[new_ids[u]],
                index[new_ids[u] + 1]); // sort neighbors of each vertex
    }
    DestID_ *inv_neighs(nullptr);
    DestID_ **inv_index(nullptr);
    if (createOnlyDegList == true || createBothCSRs == true) {
      // making the inverse list (in-degrees in this case)
      pvector<SGOffset> inv_offsets = ParallelPrefixSum(degrees);
      inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
      inv_index =
        CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
      if (createBothCSRs == true) {
#pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++) {
          if (outDegree == true) {
            for (NodeID_ v : g.out_neigh(u))
              inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
          } else {
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
    PrintTime("DegSort time", t.Seconds());
    if (outDegree == true) {
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                                                inv_neighs, index, neighs);
    } else {
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                                                inv_index, inv_neighs);
    }
  } else {
/* Undirected graphs - no need to make separate lists for in and out degree */
/* Step I: Create a list of degrees */
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
    }

    /* Step II: Sort based on degree order */
    __gnu_parallel::sort(
      degree_id_pairs.begin(), degree_id_pairs.end(),
      std::greater<degree_node_p>()); // TODO:Use parallel sort

/* Step III: assigned remap for the hub vertices */
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      new_ids[degree_id_pairs[n].second] = n;
    }

    /* Step VI: generate degree to build a new graph */
    pvector<NodeID_> degrees(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      degrees[new_ids[n]] = g.out_degree(n);
    }

    /* Graph building phase */
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (NodeID_ v : g.out_neigh(u))
        neighs[offsets[new_ids[u]]++] = new_ids[v];
      std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
    }
    t.Stop();
    PrintTime("DegSort time", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
  }
}

static CSRGraph<NodeID_, DestID_, invert>
RandOrder(const CSRGraph<NodeID_, DestID_, invert> &g,
          pvector<NodeID_> &new_ids, bool createOnlyDegList,
          bool createBothCSRs) {
  Timer t;
  t.Start();
  std::srand(0); // so that the random graph generated is the same everytime
  bool outDegree = true;

  if (g.directed() == true) {
    // Step I: create a random permutation - SLOW implementation
    pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

    // #pragma omp parallel for
    for (NodeID_ v = 0; v < g.num_nodes(); ++v) {
      while (true) {
        NodeID_ randID = std::rand() % g.num_nodes();
        if (claimedVtxs[randID] != 1) {
          if (compare_and_swap(claimedVtxs[randID], 0, 1) == true) {
            new_ids[v] = randID;
            break;
          } else
            continue;
        }
      }
    }

#pragma omp parallel for
    for (NodeID_ v = 0; v < g.num_nodes(); ++v)
      assert(new_ids[v] != -1);

    /* Step VI: generate degree to build a new graph */
    pvector<NodeID_> degrees(g.num_nodes());
    pvector<NodeID_> inv_degrees(g.num_nodes());
    if (outDegree == true) {
#pragma omp parallel for
      for (NodeID_ n = 0; n < g.num_nodes(); n++) {
        degrees[new_ids[n]] = g.out_degree(n);
        inv_degrees[new_ids[n]] = g.in_degree(n);
      }
    } else {
#pragma omp parallel for
      for (NodeID_ n = 0; n < g.num_nodes(); n++) {
        degrees[new_ids[n]] = g.in_degree(n);
        inv_degrees[new_ids[n]] = g.out_degree(n);
      }
    }

    /* Graph building phase */
    pvector<SGOffset> offsets = ParallelPrefixSum(inv_degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      if (outDegree == true) {
        for (NodeID_ v : g.in_neigh(u))
          neighs[offsets[new_ids[u]]++] = new_ids[v];
      } else {
        for (NodeID_ v : g.out_neigh(u))
          neighs[offsets[new_ids[u]]++] = new_ids[v];
      }
      std::sort(index[new_ids[u]],
                index[new_ids[u] + 1]); // sort neighbors of each vertex
    }
    DestID_ *inv_neighs(nullptr);
    DestID_ **inv_index(nullptr);
    if (createOnlyDegList == true || createBothCSRs == true) {
      // making the inverse list (in-degrees in this case)
      pvector<SGOffset> inv_offsets = ParallelPrefixSum(degrees);
      inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
      inv_index =
        CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
      if (createBothCSRs == true) {
#pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++) {
          if (outDegree == true) {
            for (NodeID_ v : g.out_neigh(u))
              inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
          } else {
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
    PrintTime("RandOrder time", t.Seconds());
    if (outDegree == true) {
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                                                inv_neighs, index, neighs);
    } else {
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                                                inv_index, inv_neighs);
    }
  } else {
    /* Undirected graphs - no need to make separate lists for in and out
     * degree */
    // Step I: create a random permutation - SLOW implementation
    pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

    // #pragma omp parallel for
    for (NodeID_ v = 0; v < g.num_nodes(); ++v) {
      while (true) {
        NodeID_ randID = std::rand() % g.num_nodes();
        if (claimedVtxs[randID] != 1) {
          if (compare_and_swap(claimedVtxs[randID], 0, 1) == true) {
            new_ids[v] = randID;
            break;
          } else
            continue;
        }
      }
    }

#pragma omp parallel for
    for (NodeID_ v = 0; v < g.num_nodes(); ++v)
      assert(new_ids[v] != -1);

    /* Step VI: generate degree to build a new graph */
    pvector<NodeID_> degrees(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      degrees[new_ids[n]] = g.out_degree(n);
    }

    /* Graph building phase */
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (NodeID_ v : g.out_neigh(u))
        neighs[offsets[new_ids[u]]++] = new_ids[v];
      std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
    }
    t.Stop();
    PrintTime("RandOrder time", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
  }
}

/*
   Return a compressed transpose matrix (Rereference Matrix)
 */
static void makeOffsetMatrix(const CSRGraph<NodeID_, DestID_, invert> &g,
                             pvector<uint8_t> &offsetMatrix,
                             int numVtxPerLine, int numEpochs,
                             bool traverseCSR = true) {
  if (g.directed() == false)
    traverseCSR = true;

  Timer tm;

  /* Step I: Collect quantized edges & Compact vertices into "super vertices"
   */
  tm.Start();
  NodeID_ numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
  NodeID_ epochSz = (g.num_nodes() + numEpochs - 1) / numEpochs;
  pvector<NodeID_> lastRef(numCacheLines * numEpochs, -1);
  NodeID_ chunkSz = 64 / numVtxPerLine;
  if (chunkSz == 0)
    chunkSz = 1;

#pragma omp parallel for schedule(dynamic, chunkSz)
  for (NodeID_ c = 0; c < numCacheLines; ++c) {
    NodeID_ startVtx = c * numVtxPerLine;
    NodeID_ endVtx = (c + 1) * numVtxPerLine;
    if (c == numCacheLines - 1)
      endVtx = g.num_nodes();

    for (NodeID_ v = startVtx; v < endVtx; ++v) {
      if (traverseCSR == true) {
        for (NodeID_ ngh : g.out_neigh(v)) {
          NodeID_ nghEpoch = ngh / epochSz;
          lastRef[(c * numEpochs) + nghEpoch] =
            std::max(ngh, lastRef[(c * numEpochs) + nghEpoch]);
        }
      } else {
        for (NodeID_ ngh : g.in_neigh(v)) {
          NodeID_ nghEpoch = ngh / epochSz;
          lastRef[(c * numEpochs) + nghEpoch] =
            std::max(ngh, lastRef[(c * numEpochs) + nghEpoch]);
        }
      }
    }
  }
  tm.Stop();
  std::cout << "[CSR-HYBRID-PREPROCESSING] Time to quantize nghs and compact "
          "vertices = "
            << tm.Seconds() << std::endl;
  assert(numEpochs == 256);

  /* Step II: Converting adjacency matrix into offsets */
  tm.Start();
  uint8_t maxReref = 127; // because MSB is reserved for identifying between
                          // reref val (1) & switch point (0)
  NodeID_ subEpochSz =
    (epochSz + 127) /
    128; // Using remaining 7 bits to identify intra-epoch information
  pvector<uint8_t> compressedOffsets(numCacheLines * numEpochs);
  uint8_t mask = 1;
  uint8_t orMask = mask << 7;
  uint8_t andMask = ~(orMask);
  assert(orMask == 128 && andMask == 127);
#pragma omp parallel for schedule(static)
  for (NodeID_ c = 0; c < numCacheLines; ++c) {
    { // first set values for the last epoch
      NodeID_ e = numEpochs - 1;
      if (lastRef[(c * numEpochs) + e] != -1) {
        compressedOffsets[(c * numEpochs) + e] = maxReref;
        compressedOffsets[(c * numEpochs) + e] &= andMask;
      } else {
        compressedOffsets[(c * numEpochs) + e] = maxReref;
        compressedOffsets[(c * numEpochs) + e] |= orMask;
      }
    }

    // Now back track and set values for all epochs
    for (NodeID_ e = numEpochs - 2; e >= 0; --e) {
      if (lastRef[(c * numEpochs) + e] != -1) {
        // There was a ref this epoch - store the quantized val of the lastRef
        NodeID_ subEpochDist = lastRef[(c * numEpochs) + e] - (e * epochSz);
        assert(subEpochDist >= 0);
        NodeID_ lastRefQ = (subEpochDist / subEpochSz);
        assert(lastRefQ <= maxReref);
        compressedOffsets[(c * numEpochs) + e] =
          static_cast<uint8_t>(lastRefQ);
        compressedOffsets[(c * numEpochs) + e] &= andMask;
      } else {
        if ((compressedOffsets[(c * numEpochs) + e + 1] & orMask) != 0) {
          // No access next epoch as well - add inter-epoch distance
          uint8_t nextRef =
            compressedOffsets[(c * numEpochs) + e + 1] & andMask;
          if (nextRef == maxReref)
            compressedOffsets[(c * numEpochs) + e] = maxReref;
          else
            compressedOffsets[(c * numEpochs) + e] = nextRef + 1;
        } else {
          // There is an access next epoch - so inter-epoch distance is set to
          // next epoch
          compressedOffsets[(c * numEpochs) + e] = 1;
        }
        compressedOffsets[(c * numEpochs) + e] |= orMask;
      }
    }
  }
  tm.Stop();
  std::cout
          << "[CSR-HYBRID-PREPROCESSING] Time to convert to offsets matrix = "
          << tm.Seconds() << std::endl;

  /* Step III: Transpose edgePresent*/
  tm.Start();
#pragma omp parallel for schedule(static)
  for (NodeID_ c = 0; c < numCacheLines; ++c) {
    for (NodeID_ e = 0; e < numEpochs; ++e) {
      offsetMatrix[(e * numCacheLines) + c] =
        compressedOffsets[(c * numEpochs) + e];
    }
  }
  tm.Stop();
  std::cout
          << "[CSR-HYBRID-PREPROCESSING] Time to transpose offsets matrix =  "
          << tm.Seconds() << std::endl;
}

//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//
typedef std::vector<std::vector<std::pair<rabbit_order::vint, float> > >
        adjacency_list;

rabbit_order::vint
count_unused_id(const rabbit_order::vint n,
                const std::vector<edge_list::edge> &edges) {
  std::vector<char> appears(n);
  for (size_t i = 0; i < edges.size(); ++i) {
    appears[std::get<0>(edges[i])] = true;
    appears[std::get<1>(edges[i])] = true;
  }
  return static_cast<rabbit_order::vint>(boost::count(appears, false));
}

template <typename RandomAccessRange>
adjacency_list make_adj_list(const rabbit_order::vint n,
                             const RandomAccessRange &es) {
  using std::get;

  // Symmetrize the edge list and remove self-loops simultaneously
  std::vector<edge_list::edge> ss(boost::size(es) * 2);
#pragma omp parallel for
  for (size_t i = 0; i < boost::size(es); ++i) {
    auto &e = es[i];
    if (get<0>(e) != get<1>(e)) {
      ss[i * 2] = std::make_tuple(get<0>(e), get<1>(e), get<2>(e));
      ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), get<2>(e));
    } else {
      // Insert zero-weight edges instead of loops; they are ignored in making
      // an adjacency list
      ss[i * 2] = std::make_tuple(0, 0, 0.0f);
      ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
    }
  }

  // Sort the edges
  __gnu_parallel::sort(ss.begin(), ss.end());

  // Convert to an adjacency list
  adjacency_list adj(n);
#pragma omp parallel
  {
    // Advance iterators to a boundary of a source vertex
    const auto adv = [](auto it, const auto first, const auto last) {
           while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
             ++it;
           return it;
         };

    // Compute an iterator range assigned to this thread
    const int p = omp_get_max_threads();
    const size_t t = static_cast<size_t>(omp_get_thread_num());
    const size_t ifirst = ss.size() / p * (t) + std::min(t, ss.size() % p);
    const size_t ilast =
      ss.size() / p * (t + 1) + std::min(t + 1, ss.size() % p);
    auto it = adv(ss.begin() + ifirst, ss.begin(), ss.end());
    const auto last = adv(ss.begin() + ilast, ss.begin(), ss.end());

    // Reduce edges and store them in std::vector
    while (it != last) {
      const rabbit_order::vint s = get<0>(*it);

      // Obtain an upper bound of degree and reserve memory
      const auto maxdeg =
        std::find_if(it, last, [s](auto &x) {
          return get<0>(x) != s;
        }) -
        it;
      adj[s].reserve(maxdeg);

      while (it != last && get<0>(*it) == s) {
        const rabbit_order::vint t = get<1>(*it);
        float w = 0.0;
        while (it != last && get<0>(*it) == s && get<1>(*it) == t)
          w += get<2>(*it++);
        if (w > 0.0)
          adj[s].push_back({t, w});
      }

      // The actual degree can be smaller than the upper bound
      adj[s].shrink_to_fit();
    }
  }

  return adj;
}

adjacency_list read_graph(const std::string &graphpath) {
  const auto edges = edge_list::read(graphpath);

  // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
  const auto n = boost::accumulate(
    edges, static_cast<rabbit_order::vint>(0),
    [](rabbit_order::vint s, auto &e) {
      return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);
    });

  if (const size_t c = count_unused_id(n, edges)) {
    // std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
    // << " (zero-degree vertices or noncontiguous IDs?)\n";
  }

  return make_adj_list(n, edges);
}

adjacency_list
readRabbitOrderGraphCSR(const CSRGraph<NodeID_, DestID_, invert> &g) {

  std::vector<edge> edges(g.num_edges(), {0, 0, 0.0f});

  if (g.directed()) {
    edges.resize(g.num_edges_directed(), {0, 0, 0.0f});
  }

  for (NodeID_ i = 0; i < g.num_nodes(); i++) {
    for (DestID_ j : g.out_neigh(i)) {
      // edges.push_back({i, j, j.w});
      edges.push_back({i, j, 1.0f});
    }
  }

  if (g.directed()) {
    for (NodeID_ i = 0; i < g.num_nodes(); i++) {
      for (DestID_ j : g.in_neigh(i)) {
        // edges.push_back({i, j, j.w});
        edges.push_back({i, j, 1.0f});
      }
    }
  }

  // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
  const auto n = boost::accumulate(
    edges, static_cast<rabbit_order::vint>(0),
    [](rabbit_order::vint s, auto &e) {
      return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);
    });

  // if (const size_t c = count_unused_id(n, edges)) {
    // std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
    //           << " (zero-degree vertices or noncontiguous IDs?)\n";
  // }

  return make_adj_list(n, edges);
}

void GenerateRabbitOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                pvector<NodeID_> &new_ids) {
  using boost::adaptors::transformed;

  // std::cerr << "Number of threads: " << omp_get_num_threads() << std::endl;
  // omp_set_num_threads(omp_get_max_threads());

  // std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;
  auto adj = readRabbitOrderGraphCSR(g);
  // const auto m =
  //     boost::accumulate(adj | transformed([](auto &es) { return es.size();
  //     }),
  //                       static_cast<size_t>(0));
  // std::cerr << "Number of vertices: " << adj.size() << std::endl;
  // std::cerr << "Number of edges: " << m << std::endl;

  // if (commode)
  //   detect_community(std::move(adj));
  // else
  reorder_internal(std::move(adj), new_ids);
}

template <typename InputIt>
typename std::iterator_traits<InputIt>::difference_type
count_uniq(const InputIt f, const InputIt l) {
  std::vector<typename std::iterator_traits<InputIt>::value_type> ys(f, l);
  return boost::size(boost::unique(boost::sort(ys)));
}

double compute_modularity(const adjacency_list &adj,
                          const rabbit_order::vint *const coms) {
  const rabbit_order::vint n = static_cast<rabbit_order::vint>(adj.size());
  const auto ncom = count_uniq(coms, coms + n);
  double m2 = 0.0; // total weight of the (bidirectional) edges

  std::unordered_map<rabbit_order::vint, double[2]> degs(
    ncom); // ID -> {all, loop}
  degs.reserve(ncom);

#pragma omp parallel reduction(+ : m2)
  {
    std::unordered_map<rabbit_order::vint, double[2]> mydegs(ncom);
    mydegs.reserve(ncom);

#pragma omp for
    for (rabbit_order::vint v = 0; v < n; ++v) {
      const rabbit_order::vint c = coms[v];
      auto *const d = &mydegs[c];
      for (const auto e : adj[v]) {
        m2 += e.second;
        (*d)[0] += e.second;
        if (coms[e.first] == c)
          (*d)[1] += e.second;
      }
    }

#pragma omp critical
    {
      for (auto &kv : mydegs) {
        auto *const d = &degs[kv.first];
        (*d)[0] += kv.second[0];
        (*d)[1] += kv.second[1];
      }
    }
  }
  assert(static_cast<intmax_t>(degs.size()) == ncom);

  double q = 0.0;
  for (auto &kv : degs) {
    const double all = kv.second[0];
    const double loop = kv.second[1];
    q += loop / m2 - (all / m2) * (all / m2);
  }

  return q;
}

void detect_community(adjacency_list adj) {
  auto _adj = adj; // copy `adj` because it is used for computing modularity

  // std::cerr << "Detecting communities...\n";
  const double tstart = rabbit_order::now_sec();
  //--------------------------------------------
  auto g = rabbit_order::aggregate(std::move(_adj));
  const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
#pragma omp parallel for
  for (rabbit_order::vint v = 0; v < g.n(); ++v)
    c[v] = rabbit_order::trace_com(v, &g);
  //--------------------------------------------
  // std::cerr << "Community detection time"
  //           << rabbit_order::now_sec() - tstart << std::endl;
  PrintTime("Community time", rabbit_order::now_sec() - tstart);

  // Print the result
  std::copy(&c[0], &c[g.n()],
            std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));

  // std::cerr << "Computing modularity of the result...\n";
  const double q = compute_modularity(adj, c.get());
  // std::cerr << "Modularity: " << q << std::endl;
}

void reorder(adjacency_list adj) {
  // std::cerr << "Generating a permutation...\n";
  const double tstart = rabbit_order::now_sec();
  //--------------------------------------------
  const auto g = rabbit_order::aggregate(std::move(adj));
  const auto p = rabbit_order::compute_perm(g);
  //--------------------------------------------
  // std::cerr << "Permutation generation time: "
  //           << rabbit_order::now_sec() - tstart << std::endl;
  PrintTime("Permutation generation time", rabbit_order::now_sec() - tstart);
  // Print the result
  std::copy(&p[0], &p[g.n()],
            std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));
}

void reorder_internal(adjacency_list adj, pvector<NodeID_> &new_ids) {
  // std::cerr << "Generating a permutation...\n";
  const double tstart = rabbit_order::now_sec();
  //--------------------------------------------
  const auto g = rabbit_order::aggregate(std::move(adj));
  const auto p = rabbit_order::compute_perm(g);
  //--------------------------------------------
  // std::cerr << "Permutation generation time: "
  //           << rabbit_order::now_sec() - tstart << std::endl;
  PrintTime("Permutation time", rabbit_order::now_sec() - tstart);
  // Ensure new_ids is large enough to hold all new IDs

  if (new_ids.size() < g.n())
    new_ids.resize(g.n());

#pragma omp parallel for
  for (size_t i = 0; i < g.n(); ++i) {
    new_ids[i] = (NodeID_)p[i];
  }
}

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

//   void GenerateGOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
//                              pvector<NodeID_> &new_ids) {

//     std::vector<std::pair<int, int>> edges(g.num_edges(), {0, 0});
//     int window = 5;

//     for (NodeID_ i = 0; i < g.num_nodes(); i++) {
//       for (DestID_ j : g.out_neigh(i)) {
//         edges.push_back({i, j});
//       }
//     }

//     if (g.directed()) {
//       for (NodeID_ i = 0; i < g.num_nodes(); i++) {
//         for (DestID_ j : g.in_neigh(i)) {
//           edges.push_back({i, j});
//         }
//       }
//     }

//     Gorder::GoGraph go;
//     vector<int> order;
//     Timer tm;
//     std::string name;
//     name = GorderUtil::extractFilename(cli_.filename().c_str());
//     go.setFilename(name);

//     tm.Start();
//     go.readGraphEdgelist(edges, g.num_nodes());
//     go.Transform();
//     tm.Stop();
//     PrintTime("Gorder graph", tm.Seconds());

//     tm.Start();
//     go.GorderGreedy(order, window);
//     tm.Stop();
//     PrintTime("Gorder time", tm.Seconds());

// #pragma omp parallel for
//     for (int i = 0; i < go.vsize; i++) {
//       int u = order[go.order_l1[i]];
//       new_ids[i] = (NodeID_)u;
//     }
//   }
};

#endif // BUILDER_H_