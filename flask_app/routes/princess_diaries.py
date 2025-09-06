from flask import Blueprint, request, jsonify
import heapq

princess_bp = Blueprint("princess", __name__)

def build_graph(subway):
    g = {}
    for edge in subway:
        u, v = edge["connection"]
        w = edge["fee"]
        g.setdefault(u, []).append((v, w))
        g.setdefault(v, []).append((u, w))
    return g

def dijkstra(g, src):
    INF = 10**18
    dist = {u: INF for u in g.keys()}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in g[u]:
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def all_pairs_to_relevant(g, relevant):
    dmap = {}
    for r in relevant:
        dist = dijkstra(g, r)
        for s in relevant:
            dmap[(r, s)] = dist[s]
    return dmap

class Task:
    def __init__(self, name, start, end, station, score, idx):
        self.name = name
        self.start = start
        self.end = end
        self.station = station
        self.score = score
        self.idx = idx

@princess_bp.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    payload = request.get_json(force=True)

    tasks_in = payload["tasks"]
    subway = payload["subway"]
    s0 = payload["starting_station"]

    g = build_graph(subway)
    g.setdefault(s0, [])

    tasks = []
    for i, t in enumerate(tasks_in):
        tasks.append(Task(
            name=t["name"],
            start=t["start"],
            end=t["end"],
            station=t["station"],
            score=t["score"],
            idx=i
        ))

    NEG_INF, POS_INF = -1, 10**9
    start_sentinel = Task("__start__", NEG_INF, NEG_INF, s0, 0, -1)
    end_sentinel   = Task("__end__", POS_INF, POS_INF+1, s0, 0, -2)

    all_tasks = [start_sentinel] + tasks + [end_sentinel]
    all_tasks.sort(key=lambda t: (t.start, t.end))

    relevant_stations = sorted({t.station for t in all_tasks} | set(g.keys()))
    D = all_pairs_to_relevant(g, relevant_stations)

    def fee(u, v): return D[(u, v)]

    n = len(all_tasks)
    preds = [[] for _ in range(n)]
    for j in range(n):
        sj = all_tasks[j].start
        for i in range(j):
            if all_tasks[i].end <= sj:
                preds[j].append(i)

    INF = 10**18
    dp_score = [-10**9] * n
    dp_fee   = [INF] * n
    parent   = [-1] * n

    start_idx = next(i for i, t in enumerate(all_tasks) if t.name == "__start__")
    dp_score[start_idx], dp_fee[start_idx] = 0, 0

    for j in range(n):
        for i in preds[j]:
            cand_score = dp_score[i] + all_tasks[j].score
            cand_fee   = dp_fee[i] + fee(all_tasks[i].station, all_tasks[j].station)
            if cand_score > dp_score[j] or (cand_score == dp_score[j] and cand_fee < dp_fee[j]):
                dp_score[j], dp_fee[j], parent[j] = cand_score, cand_fee, i

    end_idx = next(i for i, t in enumerate(all_tasks) if t.name == "__end__")

    chosen_names = []
    cur = end_idx
    while cur != -1 and cur != start_idx:
        t = all_tasks[cur]
        if t.name not in ("__start__", "__end__"):
            chosen_names.append(t.name)
        cur = parent[cur]
    chosen_names.reverse()

    name_to_task = {t.name: t for t in tasks}
    chosen_names.sort(key=lambda nm: name_to_task[nm].start)

    return jsonify({
        "max_score": int(dp_score[end_idx]),
        "min_fee": int(dp_fee[end_idx]),
        "schedule": chosen_names,
    })
