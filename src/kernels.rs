// WGSL compute kernels for FQMS mesh simplification.
// Each constant is a complete shader, compiled at runtime by wgpu.

/// Fill idx buffer with 0, 1, 2, ..., n-1
pub const SEQUENCE: &str = r#"
struct Params { n: u32 }

@group(0) @binding(0) var<storage, read_write> idx: array<i32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    idx[gid.x] = i32(gid.x);
}
"#;

/// Count how many triangles reference each vertex (via atomicAdd on tcount).
/// Launched with num_half_edges threads (num_faces * 3).
pub const UPDATE_FACES_COUNT: &str = r#"
struct Params { num_half_edges: u32 }

@group(0) @binding(0) var<storage, read_write> tri_v: array<i32>;
@group(0) @binding(1) var<storage, read_write> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> tcount: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_half_edges) { return; }
    let face_idx = idx[i / 3u];
    let local_v = i32(i % 3u);
    let vertex_idx = tri_v[face_idx * 3 + local_v];
    atomicAdd(&tcount[vertex_idx], 1);
}
"#;

/// Build the half-edge list. Each thread writes one edge entry.
/// Uses atomicAdd on tcount to get the write offset within each vertex's edge block.
/// Launched with num_half_edges threads.
pub const UPDATE_EDGE: &str = r#"
struct Params { num_half_edges: u32 }

@group(0) @binding(0) var<storage, read_write> tri_v: array<i32>;
@group(0) @binding(1) var<storage, read_write> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> tcount: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> tstart: array<i32>;
@group(0) @binding(4) var<storage, read_write> edge_tid: array<i32>;
@group(0) @binding(5) var<storage, read_write> edge_tvertex: array<i32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_half_edges) { return; }
    let face_idx = i32(i / 3u);
    let face_id = idx[face_idx];
    let local_v = i32(i % 3u);
    let vertex_idx = tri_v[face_id * 3 + local_v];
    let edge_idx = atomicAdd(&tcount[vertex_idx], 1) + tstart[vertex_idx];
    edge_tid[edge_idx] = face_idx;
    edge_tvertex[edge_idx] = local_v;
}
"#;

/// Detect boundary vertices. A vertex is on the boundary if any of its
/// neighbor vertices appears in only one adjacent triangle.
/// Launched with num_vertices threads.
pub const DETECT_BOUNDARY: &str = r#"
struct Params { num_vertices: u32 }

@group(0) @binding(0) var<storage, read_write> tcount: array<i32>;
@group(0) @binding(1) var<storage, read_write> tstart: array<i32>;
@group(0) @binding(2) var<storage, read_write> edge_tid: array<i32>;
@group(0) @binding(3) var<storage, read_write> tri_v: array<i32>;
@group(0) @binding(4) var<storage, read_write> border: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> global_vids: array<i32>;
@group(0) @binding(6) var<storage, read_write> global_vcount: array<i32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v_index = i32(gid.x);
    if (gid.x >= params.num_vertices) { return; }

    let tc = tcount[v_index];
    let ts = tstart[v_index];
    let buf_start = ts * 2 + v_index;

    var num_found: i32 = 0;

    for (var j: i32 = 0; j < tc; j += 1) {
        let triangle_id = edge_tid[ts + j];
        for (var k: i32 = 0; k < 3; k += 1) {
            let vid = tri_v[triangle_id * 3 + k];

            // linear search in found-so-far
            var offset: i32 = 0;
            while (offset < num_found && global_vids[buf_start + offset] != vid) {
                offset += 1;
            }

            if (offset == num_found) {
                // new neighbor
                global_vids[buf_start + num_found] = vid;
                global_vcount[buf_start + num_found] = 1;
                num_found += 1;
            } else {
                global_vcount[buf_start + offset] += 1;
            }
        }
    }

    // any neighbor seen only once => boundary
    for (var j: i32 = 0; j < num_found; j += 1) {
        if (global_vcount[buf_start + j] == 1) {
            atomicOr(&border[global_vids[buf_start + j]], 1);
        }
    }
}
"#;

/// Compute the QEM (quadric error metric) for each face and atomically
/// accumulate it into the 10-float QEM of each vertex.
/// Also writes face normals.
/// Launched with num_faces threads.
pub const INITIATE_Q_MAT: &str = r#"
struct Params { num_faces: u32 }

@group(0) @binding(0) var<storage, read_write> tri_v: array<i32>;
@group(0) @binding(1) var<storage, read_write> vertex_p: array<f32>;
@group(0) @binding(2) var<storage, read_write> vertex_q: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> face_n: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn atomic_add_f32(index: i32, val: f32) {
    var old = atomicLoad(&vertex_q[index]);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old) + val);
        let result = atomicCompareExchangeWeak(&vertex_q[index], old, new_val);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_idx = i32(gid.x);
    if (gid.x >= params.num_faces) { return; }

    // read 3 vertex positions
    var v: array<i32, 3>;
    var p: array<vec3<f32>, 3>;
    for (var j: i32 = 0; j < 3; j += 1) {
        v[j] = tri_v[face_idx * 3 + j];
        let b = v[j] * 3;
        p[j] = vec3<f32>(vertex_p[b], vertex_p[b + 1], vertex_p[b + 2]);
    }

    // face normal
    let e1 = p[1] - p[0];
    let e2 = p[2] - p[0];
    var n = cross(e1, e2);
    let len = length(n);
    if (len > 0.0) {
        n = n / len;
    } else {
        n = vec3<f32>(1.0, 0.0, 0.0);
    }

    // store face normal
    face_n[face_idx * 3]     = n.x;
    face_n[face_idx * 3 + 1] = n.y;
    face_n[face_idx * 3 + 2] = n.z;

    // plane equation: n.x*X + n.y*Y + n.z*Z + d = 0
    let d = -dot(n, p[0]);

    // quadric Q = [a*a, a*b, a*c, a*d, b*b, b*c, b*d, c*c, c*d, d*d]
    var q: array<f32, 10>;
    q[0] = n.x * n.x;  q[1] = n.x * n.y;  q[2] = n.x * n.z;  q[3] = n.x * d;
    q[4] = n.y * n.y;  q[5] = n.y * n.z;  q[6] = n.y * d;
    q[7] = n.z * n.z;  q[8] = n.z * d;    q[9] = d * d;

    // atomic-add Q into each vertex's QEM
    for (var j: i32 = 0; j < 3; j += 1) {
        let base = v[j] * 10;
        for (var k: i32 = 0; k < 10; k += 1) {
            atomic_add_f32(base + k, q[k]);
        }
    }
}
"#;

/// Compute the edge-collapse error for each edge of each triangle.
/// Stores 4 floats per face: err[0..2] = edge errors, err[3] = min.
/// Launched with num_faces threads.
pub const INITIATE_ERROR: &str = r#"
struct Params { num_faces: u32 }

@group(0) @binding(0) var<storage, read_write> tri_v: array<i32>;
@group(0) @binding(1) var<storage, read_write> vertex_p: array<f32>;
@group(0) @binding(2) var<storage, read_write> vertex_q: array<f32>;
@group(0) @binding(3) var<storage, read_write> vertex_border: array<i32>;
@group(0) @binding(4) var<storage, read_write> tri_err: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn get_pos(v: i32) -> vec3<f32> {
    let b = v * 3;
    return vec3<f32>(vertex_p[b], vertex_p[b + 1], vertex_p[b + 2]);
}

fn vertex_error_q(q: array<f32, 10>, x: f32, y: f32, z: f32) -> f32 {
    return q[0]*x*x + 2.0*q[1]*x*y + 2.0*q[2]*x*z + 2.0*q[3]*x
         + q[4]*y*y + 2.0*q[5]*y*z + 2.0*q[6]*y
         + q[7]*z*z + 2.0*q[8]*z + q[9];
}

fn det3_q(q: array<f32, 10>,
          a11: i32, a12: i32, a13: i32,
          a21: i32, a22: i32, a23: i32,
          a31: i32, a32: i32, a33: i32) -> f32 {
    return q[a11]*q[a22]*q[a33] + q[a13]*q[a21]*q[a32] + q[a12]*q[a23]*q[a31]
         - q[a13]*q[a22]*q[a31] - q[a11]*q[a23]*q[a32] - q[a12]*q[a21]*q[a33];
}

fn calculate_error(i0: i32, i1: i32) -> f32 {
    // combine QEMs: q = q0 + q1
    var q: array<f32, 10>;
    for (var i: i32 = 0; i < 10; i += 1) {
        q[i] = vertex_q[i0 * 10 + i] + vertex_q[i1 * 10 + i];
    }

    let is_border = (vertex_border[i0] & vertex_border[i1]) != 0;

    // check conditioning of upper-left 3x3
    let trace = q[0] + q[4] + q[7];
    let det = det3_q(q, 0, 1, 2, 1, 4, 5, 2, 5, 7);
    let k = abs(det / (trace * trace * trace));

    if (k > 1e-9 && !is_border) {
        // well-conditioned: solve for optimal vertex via Cramer's rule
        let inv_det = 1.0 / det;
        let vx = -inv_det * det3_q(q, 1, 2, 3, 4, 5, 6, 5, 7, 8);
        let vy =  inv_det * det3_q(q, 0, 2, 3, 1, 5, 6, 2, 7, 8);
        let vz = -inv_det * det3_q(q, 0, 1, 3, 1, 4, 6, 2, 5, 8);
        return vertex_error_q(q, vx, vy, vz);
    } else {
        // fallback: pick best of p0, p1, midpoint
        let p1 = get_pos(i0);
        let p2 = get_pos(i1);
        let p3 = (p1 + p2) * 0.5;

        let e1 = vertex_error_q(q, p1.x, p1.y, p1.z);
        let e2 = vertex_error_q(q, p2.x, p2.y, p2.z);
        let e3 = vertex_error_q(q, p3.x, p3.y, p3.z);

        return min(e1, min(e2, e3));
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_idx = i32(gid.x);
    if (gid.x >= params.num_faces) { return; }

    let v0 = tri_v[face_idx * 3];
    let v1 = tri_v[face_idx * 3 + 1];
    let v2 = tri_v[face_idx * 3 + 2];

    let e0 = calculate_error(v0, v1);
    let e1 = calculate_error(v1, v2);
    let e2 = calculate_error(v2, v0);

    let base = face_idx * 4;
    tri_err[base]     = e0;
    tri_err[base + 1] = e1;
    tri_err[base + 2] = e2;
    tri_err[base + 3] = min(e0, min(e1, e2));
}
"#;

// =========================================================================
// Stream compaction (copy_if) kernels
// =========================================================================

/// Create flags: flag[i] = 1 if deleted[i]==0, else 0
pub const COPY_IF_FLAG: &str = r#"
struct Params { n: u32 }

@group(0) @binding(0) var<storage, read_write> deleted: array<i32>;
@group(0) @binding(1) var<storage, read_write> flags: array<i32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    flags[gid.x] = select(0, 1, deleted[gid.x] == 0);
}
"#;

/// Scatter: if flag[i], write idx_in[i] to idx_out[offsets[i]]
pub const COPY_IF_SCATTER: &str = r#"
struct Params { n: u32 }

@group(0) @binding(0) var<storage, read_write> flags: array<i32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<i32>;
@group(0) @binding(2) var<storage, read_write> idx_in: array<i32>;
@group(0) @binding(3) var<storage, read_write> idx_out: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    if (flags[gid.x] != 0) {
        idx_out[offsets[gid.x]] = idx_in[gid.x];
    }
}
"#;

/// Compute final count = offsets[n-1] + flags[n-1]
pub const COPY_IF_COUNT: &str = r#"
struct Params { n: u32 }

@group(0) @binding(0) var<storage, read_write> flags: array<i32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<i32>;
@group(0) @binding(2) var<storage, read_write> count_out: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
    let n = i32(params.n);
    count_out[0] = offsets[n - 1] + flags[n - 1];
}
"#;

// =========================================================================
// Edge collapse kernel — the core simplification algorithm
// =========================================================================

pub const COLLAPSE_EDGE: &str = r#"
// --- Private per-thread state ---
var<private> g_adjacent: array<i32, 64>;
var<private> g_adj_count: i32;
var<private> g_adj_init_size: i32;
var<private> g_min_intersection: i32;
var<private> g_claimed: array<i32, 36>;
var<private> g_claim_count: i32;
var<private> g_lock_id: i32;

// --- Params ---
struct Params {
    num_faces: u32,
    target_face_count: u32,
    threshold: f32,
}

// --- Bindings (16 storage + 1 uniform) ---
@group(0) @binding(0)  var<storage, read_write> tri_v:          array<i32>;
@group(0) @binding(1)  var<storage, read_write> face_idx:       array<i32>;
@group(0) @binding(2)  var<storage, read_write> tri_err:        array<f32>;
@group(0) @binding(3)  var<storage, read_write> tri_deleted:    array<i32>;
@group(0) @binding(4)  var<storage, read_write> tri_dirty:      array<i32>;
@group(0) @binding(5)  var<storage, read_write> tri_n:          array<f32>;
@group(0) @binding(6)  var<storage, read_write> vertex_p:       array<f32>;
@group(0) @binding(7)  var<storage, read_write> vertex_q:       array<f32>;
@group(0) @binding(8)  var<storage, read_write> vertex_tcount:  array<i32>;
@group(0) @binding(9)  var<storage, read_write> vertex_tstart:  array<i32>;
@group(0) @binding(10) var<storage, read_write> vertex_border:  array<i32>;
@group(0) @binding(11) var<storage, read_write> vertex_lock:    array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> vertex_vpoint:  array<i32>;
@group(0) @binding(13) var<storage, read_write> edge_tid:       array<i32>;
@group(0) @binding(14) var<storage, read_write> edge_tvertex:   array<i32>;
@group(0) @binding(15) var<storage, read_write> deleted_count:  array<atomic<i32>>;
@group(0) @binding(16) var<uniform> params: Params;

// --- Helpers ---

fn get_pos(v: i32) -> vec3<f32> {
    let b = v * 3;
    return vec3<f32>(vertex_p[b], vertex_p[b + 1], vertex_p[b + 2]);
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > 0.0) { return v / len; }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn vertex_error_q(q: array<f32, 10>, x: f32, y: f32, z: f32) -> f32 {
    return q[0]*x*x + 2.0*q[1]*x*y + 2.0*q[2]*x*z + 2.0*q[3]*x
         + q[4]*y*y + 2.0*q[5]*y*z + 2.0*q[6]*y
         + q[7]*z*z + 2.0*q[8]*z + q[9];
}

fn det3_q(q: array<f32, 10>,
          a11: i32, a12: i32, a13: i32,
          a21: i32, a22: i32, a23: i32,
          a31: i32, a32: i32, a33: i32) -> f32 {
    return q[a11]*q[a22]*q[a33] + q[a13]*q[a21]*q[a32] + q[a12]*q[a23]*q[a31]
         - q[a13]*q[a22]*q[a31] - q[a11]*q[a23]*q[a32] - q[a12]*q[a21]*q[a33];
}

struct ErrorResult {
    error: f32,
    point: vec3<f32>,
}

fn calculate_error_full(i0: i32, i1: i32) -> ErrorResult {
    var q: array<f32, 10>;
    for (var i: i32 = 0; i < 10; i += 1) {
        q[i] = vertex_q[i0 * 10 + i] + vertex_q[i1 * 10 + i];
    }
    let is_border = (vertex_border[i0] & vertex_border[i1]) != 0;
    let trace = q[0] + q[4] + q[7];
    let det = det3_q(q, 0, 1, 2, 1, 4, 5, 2, 5, 7);
    let k = abs(det / (trace * trace * trace));

    var result: ErrorResult;
    if (k > 1e-9 && !is_border) {
        let inv_det = 1.0 / det;
        result.point = vec3<f32>(
            -inv_det * det3_q(q, 1, 2, 3, 4, 5, 6, 5, 7, 8),
             inv_det * det3_q(q, 0, 2, 3, 1, 5, 6, 2, 7, 8),
            -inv_det * det3_q(q, 0, 1, 3, 1, 4, 6, 2, 5, 8));
        result.error = vertex_error_q(q, result.point.x, result.point.y, result.point.z);
    } else {
        let p1 = get_pos(i0);
        let p2 = get_pos(i1);
        let p3 = (p1 + p2) * 0.5;
        let e1 = vertex_error_q(q, p1.x, p1.y, p1.z);
        let e2 = vertex_error_q(q, p2.x, p2.y, p2.z);
        let e3 = vertex_error_q(q, p3.x, p3.y, p3.z);
        result.error = min(e1, min(e2, e3));
        if (result.error == e1) { result.point = p1; }
        else if (result.error == e2) { result.point = p2; }
        else { result.point = p3; }
    }
    return result;
}

fn calculate_error(i0: i32, i1: i32) -> f32 {
    return calculate_error_full(i0, i1).error;
}

// --- Adjacency / manifold helpers ---

fn lower_bound_adj(lo_in: i32, hi_in: i32, key: i32) -> i32 {
    var lo = lo_in; var hi = hi_in;
    while (lo < hi) {
        let mid = (lo + hi) >> 1;
        if (g_adjacent[mid] < key) { lo = mid + 1; } else { hi = mid; }
    }
    return lo;
}

fn insert_unique_sorted(v: i32) -> bool {
    let lo = lower_bound_adj(0, g_adj_count, v);
    if (lo < g_adj_count && g_adjacent[lo] == v) { return false; }
    if (g_adj_count >= 64) { return true; }
    var j = g_adj_count;
    while (j > lo) { g_adjacent[j] = g_adjacent[j - 1]; j -= 1; }
    g_adjacent[lo] = v;
    g_adj_count += 1;
    return false;
}

fn insert_if_missing_from_new(num: i32) -> bool {
    let pos_old = lower_bound_adj(0, g_adj_init_size, num);
    if (pos_old == g_adj_init_size || g_adjacent[pos_old] != num) { return false; }
    let pos_new = lower_bound_adj(g_adj_init_size, g_adj_count, num);
    if (pos_new < g_adj_count && g_adjacent[pos_new] == num) { return false; }
    if (g_adj_count >= 64) { return false; }
    var i = g_adj_count;
    while (i > pos_new) { g_adjacent[i] = g_adjacent[i - 1]; i -= 1; }
    g_adjacent[pos_new] = num;
    g_adj_count += 1;
    g_min_intersection += 1;
    return false;
}

fn if_flipped_or_non_manifold(p: vec3<f32>, center: i32, i_adj: i32, first: bool) -> bool {
    g_adj_init_size = g_adj_count;
    g_min_intersection = 0;

    var v = center;
    while (v != -1) {
        let ts = vertex_tstart[v];
        let tc = vertex_tcount[v];
        for (var k: i32 = 0; k < tc; k += 1) {
            let edge_index = ts + k;
            let tri_idx = edge_tid[edge_index];
            let tri_id = face_idx[tri_idx];
            if (tri_deleted[tri_idx] != 0) { continue; }

            let s = edge_tvertex[edge_index];
            let id1 = tri_v[tri_id * 3 + ((s + 1) % 3)];
            let id2 = tri_v[tri_id * 3 + ((s + 2) % 3)];

            if (first) {
                if (insert_unique_sorted(id1)) { return true; }
                if (insert_unique_sorted(id2)) { return true; }
            } else {
                if (insert_if_missing_from_new(id1)) { return true; }
                if (insert_if_missing_from_new(id2)) { return true; }
                let is_border_pair = (vertex_border[i_adj] != 0) && (vertex_border[center] != 0);
                if (g_min_intersection > select(2, 1, is_border_pair)) { return true; }
            }

            if (id1 == i_adj || id2 == i_adj) { continue; }

            // geometric validity
            let d1 = safe_normalize(get_pos(id1) - p);
            let d2 = safe_normalize(get_pos(id2) - p);
            if (abs(dot(d1, d2)) > 0.99) { return true; }
            let n = safe_normalize(cross(d1, d2));
            let fn_vec = vec3<f32>(tri_n[tri_id * 3], tri_n[tri_id * 3 + 1], tri_n[tri_id * 3 + 2]);
            if (dot(n, fn_vec) < 0.2) { return true; }
        }
        v = vertex_vpoint[v];
    }
    return false;
}

// --- Locking ---

fn try_lock_vertex(v: i32) -> bool {
    for (var i: i32 = 0; i < g_claim_count; i += 1) {
        if (g_claimed[i] == v) { return true; }
    }
    if (g_claim_count >= 36) { return false; }
    let result = atomicCompareExchangeWeak(&vertex_lock[v], 0, g_lock_id);
    if (!result.exchanged) {
        for (var i: i32 = 0; i < g_claim_count; i += 1) {
            atomicStore(&vertex_lock[g_claimed[i]], 0);
        }
        g_claim_count = 0;
        return false;
    }
    g_claimed[g_claim_count] = v;
    g_claim_count += 1;
    return true;
}

fn lock_1ring_chain(center: i32, v0: i32, v1: i32) -> bool {
    var block = center;
    while (block != -1) {
        let ts = vertex_tstart[block];
        let tc = vertex_tcount[block];
        for (var k: i32 = 0; k < tc; k += 1) {
            let tri_id = face_idx[edge_tid[ts + k]];
            for (var j: i32 = 0; j < 3; j += 1) {
                let nbr = tri_v[tri_id * 3 + j];
                if (nbr == v0 || nbr == v1) { continue; }
                if (!try_lock_vertex(nbr)) { return false; }
            }
        }
        block = vertex_vpoint[block];
    }
    return true;
}

fn try_lock_half_edge_chain(v0: i32, v1: i32) -> bool {
    g_claim_count = 0;
    g_lock_id = v0;
    if (v0 == v1 || v0 == 0 || v1 == 0) { return false; }
    if (!try_lock_vertex(v0)) { return false; }
    if (!try_lock_vertex(v1)) { return false; }
    if (!lock_1ring_chain(v0, v0, v1)) { return false; }
    if (!lock_1ring_chain(v1, v0, v1)) { return false; }
    return true;
}

// --- Triangle update after collapse ---

fn compact_block(v: i32, from_i1: bool, i0: i32, i1: i32) {
    let start = vertex_tstart[v];
    let cnt = vertex_tcount[v];
    var write: i32 = 0;

    for (var k: i32 = 0; k < cnt; k += 1) {
        let tri_idx = edge_tid[start + k];
        let tri_id = face_idx[tri_idx];
        if (tri_deleted[tri_idx] != 0) { continue; }

        let s = edge_tvertex[start + k];
        let id1 = tri_v[tri_id * 3 + ((s + 1) % 3)];
        let id2 = tri_v[tri_id * 3 + ((s + 2) % 3)];
        let ver_rel = select(i1, i0, from_i1);

        if (id1 == ver_rel || id2 == ver_rel) {
            tri_deleted[tri_idx] = 1;
            atomicAdd(&deleted_count[0], 1);
            continue;
        }

        tri_v[tri_id * 3 + s] = i0;

        if (write != k) {
            edge_tid[start + write] = edge_tid[start + k];
            edge_tvertex[start + write] = edge_tvertex[start + k];
        }

        let rv0 = tri_v[tri_id * 3];
        let rv1 = tri_v[tri_id * 3 + 1];
        let rv2 = tri_v[tri_id * 3 + 2];
        tri_dirty[tri_idx] = 1;
        let e0 = calculate_error(rv0, rv1);
        let e1 = calculate_error(rv1, rv2);
        let e2 = calculate_error(rv2, rv0);
        tri_err[tri_id * 4]     = e0;
        tri_err[tri_id * 4 + 1] = e1;
        tri_err[tri_id * 4 + 2] = e2;
        tri_err[tri_id * 4 + 3] = min(e0, min(e1, e2));

        write += 1;
    }
    vertex_tcount[v] = write;
}

fn update_triangles(i0: i32, i1: i32) {
    var v = i0;
    while (v != -1) { compact_block(v, false, i0, i1); v = vertex_vpoint[v]; }
    v = i1;
    while (v != -1) { compact_block(v, true, i0, i1); v = vertex_vpoint[v]; }
    // splice i1 chain onto i0 tail
    var tail = i0;
    while (vertex_vpoint[tail] != -1) { tail = vertex_vpoint[tail]; }
    vertex_vpoint[tail] = i1;
}

// --- Main kernel ---

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (gid.x >= params.num_faces) { return; }

    let remaining = i32(params.num_faces) - atomicLoad(&deleted_count[0]);
    if (remaining <= i32(params.target_face_count)) { return; }

    let fid = face_idx[idx];
    if (tri_err[fid * 4 + 3] > params.threshold) { return; }
    if (tri_deleted[idx] != 0 || tri_dirty[idx] != 0) { return; }

    for (var j: i32 = 0; j < 3; j += 1) {
        if (tri_err[fid * 4 + j] >= params.threshold) { continue; }

        let i0 = tri_v[fid * 3 + j];
        let i1 = tri_v[fid * 3 + ((j + 1) % 3)];
        if (vertex_tcount[i0] == 0 || vertex_tcount[i1] == 0) { continue; }
        if (vertex_border[i0] != vertex_border[i1]) { continue; }

        let err_result = calculate_error_full(i0, i1);

        g_adj_count = 0;
        if (if_flipped_or_non_manifold(err_result.point, i0, i1, true)) { continue; }
        if (if_flipped_or_non_manifold(err_result.point, i1, i0, false)) { continue; }
        if (!try_lock_half_edge_chain(i0, i1)) { continue; }

        // --- perform collapse ---
        let b = i0 * 3;
        vertex_p[b]     = err_result.point.x;
        vertex_p[b + 1] = err_result.point.y;
        vertex_p[b + 2] = err_result.point.z;

        for (var k: i32 = 0; k < 10; k += 1) {
            vertex_q[i0 * 10 + k] += vertex_q[i1 * 10 + k];
        }

        update_triangles(i0, i1);
        break;
    }
}
"#;

// =========================================================================
// Compaction kernels (final mesh output)
// =========================================================================

/// Assign new contiguous IDs to surviving vertices, copy positions to o_p.
pub const COMPACT_VERTICES: &str = r#"
struct Params { num_vertices: u32 }

@group(0) @binding(0) var<storage, read_write> vertex_tcount: array<i32>;
@group(0) @binding(1) var<storage, read_write> vertex_remap: array<i32>;
@group(0) @binding(2) var<storage, read_write> vertex_p: array<f32>;
@group(0) @binding(3) var<storage, read_write> vertex_o_p: array<f32>;
@group(0) @binding(4) var<storage, read_write> new_count: array<atomic<i32>>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (gid.x >= params.num_vertices) { return; }

    if (vertex_tcount[i] > 0) {
        let dst = atomicAdd(&new_count[0], 1);
        vertex_remap[i] = dst;
        let s = i * 3;
        let d = dst * 3;
        vertex_o_p[d]     = vertex_p[s];
        vertex_o_p[d + 1] = vertex_p[s + 1];
        vertex_o_p[d + 2] = vertex_p[s + 2];
    } else {
        vertex_remap[i] = -1;
    }
}
"#;

/// Remap triangle vertex indices using remap table, write to o_v.
pub const COMPACT_TRIANGLES: &str = r#"
struct Params { num_faces: u32 }

@group(0) @binding(0) var<storage, read_write> face_idx: array<i32>;
@group(0) @binding(1) var<storage, read_write> tri_v: array<i32>;
@group(0) @binding(2) var<storage, read_write> vertex_remap: array<i32>;
@group(0) @binding(3) var<storage, read_write> tri_o_v: array<i32>;
@group(0) @binding(4) var<storage, read_write> new_count: array<atomic<i32>>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (gid.x >= params.num_faces) { return; }

    let tid = face_idx[idx];
    let base = tid * 3;
    let a = tri_v[base];
    let b = tri_v[base + 1];
    let c = tri_v[base + 2];

    if (vertex_remap[a] < 0 || vertex_remap[b] < 0 || vertex_remap[c] < 0) { return; }

    let dst = atomicAdd(&new_count[0], 1);
    let out = dst * 3;
    tri_o_v[out]     = vertex_remap[a];
    tri_o_v[out + 1] = vertex_remap[b];
    tri_o_v[out + 2] = vertex_remap[c];
}
"#;
