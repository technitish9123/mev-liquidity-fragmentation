"""
Converts Uniswap V3 pool snapshots into NetworkX graphs and computes
topology metrics: modularity Q, spectral gap λ₂, effective connectivity,
average path length, giant component fraction, average effective resistance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C
import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
import logging

from networkx.algorithms.community import louvain_communities, modularity as nx_modularity

log = logging.getLogger(__name__)


def build_liquidity_graph(pools: list[dict]) -> nx.Graph:
    """
    Build weighted undirected graph from pool list.

    Nodes: token addresses (string)
    Node attributes: symbol (from token0/token1)
    Edges: one per unique token pair
    Edge weight: sum of totalValueLockedUSD across all pools for that pair

    Filters: skip pools where totalValueLockedUSD < C.UNISWAP_MIN_TVL_USD
    Multi-pools: if token0+token1 pair appears multiple times (different fee tiers),
                 aggregate by SUMMING their TVL into one edge.

    Returns nx.Graph. Returns empty graph if pools is empty.
    """
    G = nx.Graph()

    if not pools:
        return G

    # edge_data[(addr0, addr1)] = (symbol0, symbol1, cumulative_tvl)
    edge_data: dict[tuple[str, str], dict] = {}

    for pool in pools:
        try:
            tvl = float(pool.get("totalValueLockedUSD", 0.0) or 0.0)
        except (ValueError, TypeError):
            tvl = 0.0

        if tvl < C.UNISWAP_MIN_TVL_USD:
            continue

        token0 = pool.get("token0", {})
        token1 = pool.get("token1", {})

        addr0 = (token0.get("id") or token0.get("address") or "").lower().strip()
        addr1 = (token1.get("id") or token1.get("address") or "").lower().strip()
        sym0 = token0.get("symbol", addr0[:8] if addr0 else "UNK")
        sym1 = token1.get("symbol", addr1[:8] if addr1 else "UNK")

        if not addr0 or not addr1 or addr0 == addr1:
            continue

        # Canonical ordering for undirected edge key
        if addr0 < addr1:
            key = (addr0, addr1)
            s0, s1 = sym0, sym1
        else:
            key = (addr1, addr0)
            s0, s1 = sym1, sym0

        if key in edge_data:
            edge_data[key]["tvl"] += tvl
        else:
            edge_data[key] = {"sym0": s0, "sym1": s1, "tvl": tvl}

    for (addr0, addr1), info in edge_data.items():
        # Add nodes with symbol attributes (update if already present)
        if not G.has_node(addr0):
            G.add_node(addr0, symbol=info["sym0"])
        if not G.has_node(addr1):
            G.add_node(addr1, symbol=info["sym1"])

        if G.has_edge(addr0, addr1):
            G[addr0][addr1]["weight"] += info["tvl"]
        else:
            G.add_edge(addr0, addr1, weight=info["tvl"])

    log.debug(
        "build_liquidity_graph: %d pools → %d nodes, %d edges",
        len(pools), G.number_of_nodes(), G.number_of_edges(),
    )
    return G


def get_lcc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a subgraph."""
    if G.number_of_nodes() == 0:
        return G
    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()


def compute_modularity(G: nx.Graph) -> float:
    """
    Louvain community detection + modularity score.
    Uses nx.community.louvain_communities(G, weight='weight', seed=C.SEED)
    then nx.community.modularity(G, communities, weight='weight').
    Returns 0.0 if graph has < 2 nodes.
    """
    if G.number_of_nodes() < 2:
        return 0.0

    try:
        communities = louvain_communities(G, weight="weight", seed=C.SEED)
        if not communities:
            return 0.0
        Q = nx_modularity(G, communities, weight="weight")
        # modularity is defined in [-0.5, 1]; clamp to [0, 1] for safety
        return float(max(0.0, Q))
    except Exception as exc:
        log.warning("compute_modularity failed: %s", exc)
        return 0.0


def compute_spectral_gap(G: nx.Graph) -> float:
    """
    λ₂ of normalized Laplacian of the LCC.
    Use scipy.sparse.linalg.eigsh with k=min(3, n-1), which='SM'.
    Sort eigenvalues; λ₂ is index 1 (λ₁=0).
    Returns 0.0 if LCC has < 3 nodes.
    Handle LinAlgError gracefully → return 0.0.
    """
    lcc = get_lcc(G)
    n = lcc.number_of_nodes()

    if n < 3:
        return 0.0

    try:
        L = nx.normalized_laplacian_matrix(lcc, weight="weight").astype(float)
        k = min(3, n - 1)
        # which='SM' finds the k smallest-magnitude eigenvalues
        eigenvalues, _ = spla.eigsh(L, k=k, which="SM")
        eigenvalues = np.sort(np.abs(eigenvalues.real))
        if len(eigenvalues) < 2:
            return 0.0
        return float(eigenvalues[1])
    except np.linalg.LinAlgError as exc:
        log.warning("compute_spectral_gap LinAlgError: %s", exc)
        return 0.0
    except Exception as exc:
        log.warning("compute_spectral_gap failed: %s", exc)
        return 0.0


def compute_eff_connectivity(G: nx.Graph, w_min: float = None) -> float:
    """
    Fraction of node pairs (u,v) where bottleneck edge weight on the
    max-weight-bottleneck path ≥ w_min.
    w_min defaults to C.W_MIN.

    For efficiency: if |V| > 100, sample 500 random pairs instead of all pairs.
    Uses nx.maximum_spanning_tree then checks path min weight.
    Returns fraction in [0,1]. Returns 0.0 if < 2 nodes.
    """
    if w_min is None:
        w_min = C.W_MIN

    if G.number_of_nodes() < 2:
        return 0.0

    try:
        # Maximum spanning tree gives us the max-bottleneck paths
        mst = nx.maximum_spanning_tree(G, weight="weight")
        nodes = list(G.nodes())
        n = len(nodes)

        rng = np.random.default_rng(C.SEED)

        if n > 100:
            # Sample 500 random pairs
            n_pairs = 500
            idx_u = rng.integers(0, n, size=n_pairs)
            idx_v = rng.integers(0, n, size=n_pairs)
            # Avoid self-pairs
            pairs = [(nodes[u], nodes[v]) for u, v in zip(idx_u, idx_v) if u != v]
            if not pairs:
                return 0.0
        else:
            pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
            if not pairs:
                return 0.0

        connected = 0
        total = 0

        for u, v in pairs:
            total += 1
            try:
                path = nx.shortest_path(mst, source=u, target=v)
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue

            # Compute min edge weight along path (bottleneck)
            if len(path) < 2:
                continue
            bottleneck = min(
                mst[path[i]][path[i + 1]].get("weight", 0.0)
                for i in range(len(path) - 1)
            )
            if bottleneck >= w_min:
                connected += 1

        return float(connected / total) if total > 0 else 0.0

    except Exception as exc:
        log.warning("compute_eff_connectivity failed: %s", exc)
        return 0.0


def compute_avg_path_length(G: nx.Graph) -> float:
    """
    Average unweighted shortest path length in the LCC.
    Returns nx.average_shortest_path_length(lcc, weight=None).
    Returns 0.0 if LCC has < 2 nodes.
    """
    lcc = get_lcc(G)

    if lcc.number_of_nodes() < 2:
        return 0.0

    try:
        return float(nx.average_shortest_path_length(lcc, weight=None))
    except nx.NetworkXError as exc:
        log.warning("compute_avg_path_length NetworkXError: %s", exc)
        return 0.0
    except Exception as exc:
        log.warning("compute_avg_path_length failed: %s", exc)
        return 0.0


def compute_avg_eff_resistance(G: nx.Graph) -> float:
    """
    Average effective resistance across all node pairs.
    R_eff(u,v) = L†[u,u] + L†[v,v] − 2·L†[u,v] where L† = pseudoinverse of Laplacian.
    Average = (2/(n(n-1))) * Σ R_eff(u,v)
    Simplified: avg_R = (2/(n-1)) * trace(L†)
    Use np.linalg.pinv on dense Laplacian of LCC.
    Cap at 50 nodes (skip if LCC > 50 — too slow for paper deadline, return 0.0).
    Returns 0.0 if < 2 nodes.
    """
    lcc = get_lcc(G)
    n = lcc.number_of_nodes()

    if n < 2:
        return 0.0

    if n > 50:
        log.debug("compute_avg_eff_resistance: LCC too large (%d > 50), skipping", n)
        return 0.0

    try:
        L_sparse = nx.laplacian_matrix(lcc, weight="weight").astype(float)
        L_dense = L_sparse.toarray()
        L_pinv = np.linalg.pinv(L_dense)
        trace_Lpinv = np.trace(L_pinv)
        avg_R = (2.0 / (n - 1)) * trace_Lpinv
        return float(avg_R)
    except np.linalg.LinAlgError as exc:
        log.warning("compute_avg_eff_resistance LinAlgError: %s", exc)
        return 0.0
    except Exception as exc:
        log.warning("compute_avg_eff_resistance failed: %s", exc)
        return 0.0


def compute_topology_metrics(pools: list[dict]) -> dict:
    """
    Master function. Given pools list, builds graph and returns all metrics.

    Returns dict with keys:
      modularity, spectral_gap, eff_connectivity, avg_path_length,
      giant_component_pct, avg_eff_resistance, nodes, edges, active_pools

    On any error, returns safe fallback dict with zeros.
    """
    fallback = {
        "modularity": 0.0,
        "spectral_gap": 0.0,
        "eff_connectivity": 0.0,
        "avg_path_length": 0.0,
        "giant_component_pct": 0.0,
        "avg_eff_resistance": 0.0,
        "nodes": 0,
        "edges": 0,
        "active_pools": 0,
    }

    try:
        G = build_liquidity_graph(pools)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Count active pools (those that passed TVL filter and contributed edges)
        active_pools = sum(
            1 for p in pools
            if float(p.get("totalValueLockedUSD", 0.0) or 0.0) >= C.UNISWAP_MIN_TVL_USD
        )

        # Giant component fraction
        if n_nodes > 0:
            lcc = get_lcc(G)
            giant_component_pct = float(lcc.number_of_nodes() / n_nodes)
        else:
            giant_component_pct = 0.0

        modularity = compute_modularity(G)
        spectral_gap = compute_spectral_gap(G)
        eff_connectivity = compute_eff_connectivity(G)
        avg_path_length = compute_avg_path_length(G)
        avg_eff_resistance = compute_avg_eff_resistance(G)

        return {
            "modularity": modularity,
            "spectral_gap": spectral_gap,
            "eff_connectivity": eff_connectivity,
            "avg_path_length": avg_path_length,
            "giant_component_pct": giant_component_pct,
            "avg_eff_resistance": avg_eff_resistance,
            "nodes": n_nodes,
            "edges": n_edges,
            "active_pools": active_pools,
        }

    except Exception as exc:
        log.error("compute_topology_metrics failed: %s", exc, exc_info=True)
        return fallback
