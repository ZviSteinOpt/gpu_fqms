use forge_cv::{ComputeContext, GpuMatrix, Mat, PrefixScan, ScalarType};
use std::sync::Once;

mod kernels;

pub struct Vertex {
    pub p: GpuMatrix,       // Float32, ch=3  [N×1]  positions
    pub o_p: GpuMatrix,     // Float32, ch=3  [N×1]  output positions
    pub tstart: GpuMatrix,  // Int32,   ch=1  [N×1]  edge-list start
    pub tcount: GpuMatrix,  // Int32,   ch=1  [N×1]  edge-list count
    pub remap: GpuMatrix,   // Int32,   ch=1  [N×1]  compaction remap
    pub border: GpuMatrix,  // Int32,   ch=1  [N×1]  boundary flag
    pub q: GpuMatrix,       // Float32, ch=1  [N×10] QEM matrices
    pub lock: GpuMatrix,    // Int32,   ch=1  [N×1]  atomic lock
    pub vpoint: GpuMatrix,  // Int32,   ch=1  [N×1]  overflow chain
}

pub struct Triangle {
    pub v: GpuMatrix,       // Int32,   ch=1  [3N×1]  vertex indices
    pub o_v: GpuMatrix,     // Int32,   ch=1  [3N×1]  output indices
    pub err: GpuMatrix,     // Float32, ch=4  [N×1]   edge errors
    pub deleted: GpuMatrix, // Int32,   ch=1  [N×1]   deletion flag
    pub dirty: GpuMatrix,   // Int32,   ch=1  [N×1]   dirty flag
    pub n: GpuMatrix,       // Float32, ch=3  [N×1]   face normals
    pub idx: GpuMatrix,     // Int32,   ch=1  [N×1]   compacted index
}

pub struct Edge {
    pub tid: GpuMatrix,     // Int32, ch=1  [1×3N]  triangle ID
    pub tvertex: GpuMatrix, // Int32, ch=1  [1×3N]  local vertex pos
}

pub struct Fqms {
    pub vertices: Vertex,
    pub triangles: Triangle,
    pub edges: Edge,
    num_vertices: usize,
    num_faces: usize,
    target_face_count: usize,
    max_error_allowed: f32,
    aggressiveness: i32,
    scan: PrefixScan,
    // atomic counter on GPU
    deleted_triangles: GpuMatrix,
    // result counts (set by compact_mesh)
    new_num_faces: usize,
    new_num_vertices: usize,
    // copy_if scratch buffers
    cif_flags: GpuMatrix,
    cif_offsets: GpuMatrix,
    cif_temp: GpuMatrix,
    cif_count: GpuMatrix,
}

impl Vertex {
    fn new() -> Self {
        Vertex {
            p: GpuMatrix::new(), o_p: GpuMatrix::new(), tstart: GpuMatrix::new(),
            tcount: GpuMatrix::new(), remap: GpuMatrix::new(), border: GpuMatrix::new(),
            q: GpuMatrix::new(), lock: GpuMatrix::new(), vpoint: GpuMatrix::new(),
        }
    }
}

impl Triangle {
    fn new() -> Self {
        Triangle {
            v: GpuMatrix::new(), o_v: GpuMatrix::new(), err: GpuMatrix::new(),
            deleted: GpuMatrix::new(), dirty: GpuMatrix::new(), n: GpuMatrix::new(),
            idx: GpuMatrix::new(),
        }
    }
}

impl Edge {
    fn new() -> Self { Edge { tid: GpuMatrix::new(), tvertex: GpuMatrix::new() } }
}

// ---------------------------------------------------------------------------
// Kernel loading
// ---------------------------------------------------------------------------

static KERNELS_LOADED: Once = Once::new();

fn load_all_kernels() {
    KERNELS_LOADED.call_once(|| {
        ComputeContext::load_kernel("fqms_sequence", kernels::SEQUENCE);
        ComputeContext::load_kernel("fqms_update_faces_count", kernels::UPDATE_FACES_COUNT);
        ComputeContext::load_kernel("fqms_update_edge", kernels::UPDATE_EDGE);
        ComputeContext::load_kernel("fqms_detect_boundary", kernels::DETECT_BOUNDARY);
        ComputeContext::load_kernel("fqms_initiate_q_mat", kernels::INITIATE_Q_MAT);
        ComputeContext::load_kernel("fqms_initiate_error", kernels::INITIATE_ERROR);
        ComputeContext::load_kernel("fqms_copy_if_flag", kernels::COPY_IF_FLAG);
        ComputeContext::load_kernel("fqms_copy_if_scatter", kernels::COPY_IF_SCATTER);
        ComputeContext::load_kernel("fqms_copy_if_count", kernels::COPY_IF_COUNT);
        ComputeContext::load_kernel("fqms_collapse_edge", kernels::COLLAPSE_EDGE);
        ComputeContext::load_kernel("fqms_compact_vertices", kernels::COMPACT_VERTICES);
        ComputeContext::load_kernel("fqms_compact_triangles", kernels::COMPACT_TRIANGLES);
    });
}

fn wg(n: usize) -> u32 { ((n + 255) / 256) as u32 }

// ---------------------------------------------------------------------------
// FQMS implementation
// ---------------------------------------------------------------------------

impl Fqms {
    pub fn new() -> Self {
        Fqms {
            vertices: Vertex::new(),
            triangles: Triangle::new(),
            edges: Edge::new(),
            num_vertices: 0,
            num_faces: 0,
            target_face_count: 0,
            max_error_allowed: 100.0,
            aggressiveness: 5,
            scan: PrefixScan::new(),
            deleted_triangles: GpuMatrix::new(),
            new_num_faces: 0,
            new_num_vertices: 0,
            cif_flags: GpuMatrix::new(),
            cif_offsets: GpuMatrix::new(),
            cif_temp: GpuMatrix::new(),
            cif_count: GpuMatrix::new(),
        }
    }

    pub fn num_vertices(&self) -> usize { self.num_vertices }
    pub fn num_faces(&self) -> usize { self.num_faces }
    pub fn result_num_vertices(&self) -> usize { self.new_num_vertices }
    pub fn result_num_faces(&self) -> usize { self.new_num_faces }

    // ===================================================================
    // Upload + init
    // ===================================================================

    pub fn upload(&mut self, faces_mat: &Mat, vertices_mat: &Mat) {
        self.num_vertices = vertices_mat.rows;
        self.num_faces = faces_mat.rows;

        self.init_vertices(vertices_mat);
        self.init_faces(faces_mat);
        self.init_edges();
        self.init_scratch();
        self.init_mesh_data();
    }

    fn init_vertices(&mut self, vertices_mat: &Mat) {
        let n = self.num_vertices;
        self.vertices.p.upload(vertices_mat);
        self.vertices.o_p = ComputeContext::malloc(n, 1, ScalarType::Float32, 3);
        self.vertices.q = ComputeContext::malloc(n, 10, ScalarType::Float32, 1);
        self.vertices.tcount = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.vertices.tstart = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.vertices.remap = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.vertices.border = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.vertices.lock = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.vertices.vpoint = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);

        self.vertices.tcount.set_to_zero();
        self.vertices.border.set_to_zero();
        self.vertices.q.set_to_zero();
        self.vertices.lock.set_to_zero();
        self.vertices.vpoint.set_to(0xFF); // -1
    }

    fn init_faces(&mut self, faces_mat: &Mat) {
        let n = self.num_faces;
        self.extract_xyz_faces(faces_mat);
        self.triangles.deleted = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.triangles.o_v = ComputeContext::malloc(3 * n, 1, ScalarType::Int32, 1);
        self.triangles.dirty = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.triangles.n = ComputeContext::malloc(n, 1, ScalarType::Float32, 3);
        self.triangles.err = ComputeContext::malloc(n, 1, ScalarType::Float32, 4);
        self.triangles.idx = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);
        self.triangles.deleted.set_to_zero();
        self.triangles.dirty.set_to_zero();
    }

    fn extract_xyz_faces(&mut self, faces_mat: &Mat) {
        let n = self.num_faces;
        let src: &[i32] = faces_mat.as_slice();
        let mut xyz = Vec::with_capacity(3 * n);
        for i in 0..n {
            let b = i * 4;
            xyz.push(src[b]);
            xyz.push(src[b + 1]);
            xyz.push(src[b + 2]);
        }
        let m = Mat::from_slice("xyz_faces", 3 * n, 1, ScalarType::Int32, 1, &xyz);
        self.triangles.v.upload(&m);
    }

    fn init_edges(&mut self) {
        let n = self.num_faces;
        self.edges.tid = ComputeContext::malloc(1, 3 * n, ScalarType::Int32, 1);
        self.edges.tvertex = ComputeContext::malloc(1, 3 * n, ScalarType::Int32, 1);
    }

    fn init_scratch(&mut self) {
        let nf = self.num_faces;
        self.deleted_triangles = ComputeContext::malloc(1, 1, ScalarType::Int32, 1);
        self.deleted_triangles.set_to_zero();
        self.cif_flags = ComputeContext::malloc(nf, 1, ScalarType::Int32, 1);
        self.cif_offsets = ComputeContext::malloc(nf, 1, ScalarType::Int32, 1);
        self.cif_temp = ComputeContext::malloc(nf, 1, ScalarType::Int32, 1);
        self.cif_count = ComputeContext::malloc(1, 1, ScalarType::Int32, 1);
    }

    // ===================================================================
    // GPU init pipeline (mirrors CUDA init_fqms_mesh_data)
    // ===================================================================

    fn init_mesh_data(&mut self) {
        load_all_kernels();
        let nf = self.num_faces;
        let nv = self.num_vertices;
        let nhe = nf * 3;

        let p_nf = (nf as u32).to_le_bytes();
        ComputeContext::dispatch("fqms_sequence", wg(nf),
            &[self.triangles.idx.wgpu_buffer()], Some(&p_nf));

        let p_nhe = (nhe as u32).to_le_bytes();
        ComputeContext::dispatch("fqms_update_faces_count", wg(nhe), &[
            self.triangles.v.wgpu_buffer(),
            self.triangles.idx.wgpu_buffer(),
            self.vertices.tcount.wgpu_buffer(),
        ], Some(&p_nhe));

        self.scan.exclusive_scan(&self.vertices.tcount, &self.vertices.tstart);
        self.vertices.tcount.set_to_zero();

        ComputeContext::dispatch("fqms_update_edge", wg(nhe), &[
            self.triangles.v.wgpu_buffer(),
            self.triangles.idx.wgpu_buffer(),
            self.vertices.tcount.wgpu_buffer(),
            self.vertices.tstart.wgpu_buffer(),
            self.edges.tid.wgpu_buffer(),
            self.edges.tvertex.wgpu_buffer(),
        ], Some(&p_nhe));

        let temp_size = nhe * 2 + nv;
        let gv = ComputeContext::malloc(temp_size, 1, ScalarType::Int32, 1);
        let gc = ComputeContext::malloc(temp_size, 1, ScalarType::Int32, 1);
        let p_nv = (nv as u32).to_le_bytes();
        ComputeContext::dispatch("fqms_detect_boundary", wg(nv), &[
            self.vertices.tcount.wgpu_buffer(),
            self.vertices.tstart.wgpu_buffer(),
            self.edges.tid.wgpu_buffer(),
            self.triangles.v.wgpu_buffer(),
            self.vertices.border.wgpu_buffer(),
            gv.wgpu_buffer(),
            gc.wgpu_buffer(),
        ], Some(&p_nv));

        ComputeContext::dispatch("fqms_initiate_q_mat", wg(nf), &[
            self.triangles.v.wgpu_buffer(),
            self.vertices.p.wgpu_buffer(),
            self.vertices.q.wgpu_buffer(),
            self.triangles.n.wgpu_buffer(),
        ], Some(&p_nf));

        ComputeContext::dispatch("fqms_initiate_error", wg(nf), &[
            self.triangles.v.wgpu_buffer(),
            self.vertices.p.wgpu_buffer(),
            self.vertices.q.wgpu_buffer(),
            self.vertices.border.wgpu_buffer(),
            self.triangles.err.wgpu_buffer(),
        ], Some(&p_nf));
    }

    // ===================================================================
    // Simplify
    // ===================================================================

    pub fn simplify(&mut self, target_face_count: usize) {
        assert!(target_face_count > 0 && self.num_vertices >= 3 && self.num_faces > 0);
        self.target_face_count = target_face_count;

        self.triangles.deleted.set_to_zero();
        self.triangles.dirty.set_to_zero();
        self.deleted_triangles.set_to_zero();

        const MAX_ITERATIONS: usize = 200;
        for iteration in 0..MAX_ITERATIONS {
            if self.collapse(iteration) {
                break;
            }
            self.update_surface();
        }
    }

    // ===================================================================
    // Collapse — dispatches edge collapse kernel
    // ===================================================================

    fn collapse(&mut self, global_itr: usize) -> bool {
        let inner_itr = 5;

        for _ in 0..inner_itr {
            let mut threshold =
                1e-9_f32 * (global_itr as f32 + 3.0).powi(self.aggressiveness);

            let mut params = [0u8; 16];
            params[0..4].copy_from_slice(&(self.num_faces as u32).to_le_bytes());
            params[4..8].copy_from_slice(&(self.target_face_count as u32).to_le_bytes());
            params[8..12].copy_from_slice(&threshold.to_le_bytes());

            ComputeContext::dispatch("fqms_collapse_edge", wg(self.num_faces), &[
                self.triangles.v.wgpu_buffer(),        // 0
                self.triangles.idx.wgpu_buffer(),      // 1
                self.triangles.err.wgpu_buffer(),      // 2
                self.triangles.deleted.wgpu_buffer(),  // 3
                self.triangles.dirty.wgpu_buffer(),    // 4
                self.triangles.n.wgpu_buffer(),        // 5
                self.vertices.p.wgpu_buffer(),         // 6
                self.vertices.q.wgpu_buffer(),         // 7
                self.vertices.tcount.wgpu_buffer(),    // 8
                self.vertices.tstart.wgpu_buffer(),    // 9
                self.vertices.border.wgpu_buffer(),    // 10
                self.vertices.lock.wgpu_buffer(),      // 11
                self.vertices.vpoint.wgpu_buffer(),    // 12
                self.edges.tid.wgpu_buffer(),          // 13
                self.edges.tvertex.wgpu_buffer(),      // 14
                self.deleted_triangles.wgpu_buffer(),  // 15
            ], Some(&params));

            let deleted_count = self.read_i32(&self.deleted_triangles);

            if threshold > self.max_error_allowed {
                threshold = self.max_error_allowed;
            }

            if threshold == self.max_error_allowed
                || (self.num_faces as i32 - deleted_count) <= self.target_face_count as i32
            {
                self.compact_mesh();
                return true;
            }

            self.vertices.lock.set_to_zero();
            self.triangles.dirty.set_to_zero();
        }

        false
    }

    // ===================================================================
    // Update surface — stream compact + rebuild edges
    // ===================================================================

    fn update_surface(&mut self) {
        self.edges.tid.set_to_zero();
        self.edges.tvertex.set_to_zero();
        self.vertices.tcount.set_to_zero();
        self.vertices.tstart.set_to_zero();
        self.vertices.vpoint.set_to(0xFF);

        self.num_faces = self.copy_if_not_deleted();

        let nhe = self.num_faces * 3;
        let p_nhe = (nhe as u32).to_le_bytes();

        ComputeContext::dispatch("fqms_update_faces_count", wg(nhe), &[
            self.triangles.v.wgpu_buffer(),
            self.triangles.idx.wgpu_buffer(),
            self.vertices.tcount.wgpu_buffer(),
        ], Some(&p_nhe));

        self.scan.exclusive_scan(&self.vertices.tcount, &self.vertices.tstart);
        self.vertices.tcount.set_to_zero();

        ComputeContext::dispatch("fqms_update_edge", wg(nhe), &[
            self.triangles.v.wgpu_buffer(),
            self.triangles.idx.wgpu_buffer(),
            self.vertices.tcount.wgpu_buffer(),
            self.vertices.tstart.wgpu_buffer(),
            self.edges.tid.wgpu_buffer(),
            self.edges.tvertex.wgpu_buffer(),
        ], Some(&p_nhe));

        self.triangles.deleted.set_to_zero();
        self.deleted_triangles.set_to_zero();
    }

    // ===================================================================
    // Compact mesh — final vertex/face renumbering
    // ===================================================================

    fn compact_mesh(&mut self) {
        self.num_faces = self.copy_if_not_deleted();

        let nf = self.num_faces;
        let nv = self.num_vertices;
        let nhe = nf * 3;

        self.vertices.tcount.set_to_zero();
        let p_nhe = (nhe as u32).to_le_bytes();
        ComputeContext::dispatch("fqms_update_faces_count", wg(nhe), &[
            self.triangles.v.wgpu_buffer(),
            self.triangles.idx.wgpu_buffer(),
            self.vertices.tcount.wgpu_buffer(),
        ], Some(&p_nhe));

        let new_nv = ComputeContext::malloc(1, 1, ScalarType::Int32, 1);
        new_nv.set_to_zero();
        let p_nv = (nv as u32).to_le_bytes();
        ComputeContext::dispatch("fqms_compact_vertices", wg(nv), &[
            self.vertices.tcount.wgpu_buffer(),
            self.vertices.remap.wgpu_buffer(),
            self.vertices.p.wgpu_buffer(),
            self.vertices.o_p.wgpu_buffer(),
            new_nv.wgpu_buffer(),
        ], Some(&p_nv));

        let new_nf = ComputeContext::malloc(1, 1, ScalarType::Int32, 1);
        new_nf.set_to_zero();
        let p_nf = (nf as u32).to_le_bytes();
        ComputeContext::dispatch("fqms_compact_triangles", wg(nf), &[
            self.triangles.idx.wgpu_buffer(),
            self.triangles.v.wgpu_buffer(),
            self.vertices.remap.wgpu_buffer(),
            self.triangles.o_v.wgpu_buffer(),
            new_nf.wgpu_buffer(),
        ], Some(&p_nf));

        self.new_num_vertices = self.read_i32(&new_nv) as usize;
        self.new_num_faces = self.read_i32(&new_nf) as usize;
    }

    // ===================================================================
    // copy_if — stream compaction of idx where deleted==0
    // ===================================================================

    fn copy_if_not_deleted(&mut self) -> usize {
        let n = self.num_faces;
        let p_n = (n as u32).to_le_bytes();

        self.cif_flags.set_to_zero();

        ComputeContext::dispatch("fqms_copy_if_flag", wg(n), &[
            self.triangles.deleted.wgpu_buffer(),
            self.cif_flags.wgpu_buffer(),
        ], Some(&p_n));

        self.scan.exclusive_scan(&self.cif_flags, &self.cif_offsets);

        ComputeContext::dispatch("fqms_copy_if_scatter", wg(n), &[
            self.cif_flags.wgpu_buffer(),
            self.cif_offsets.wgpu_buffer(),
            self.triangles.idx.wgpu_buffer(),
            self.cif_temp.wgpu_buffer(),
        ], Some(&p_n));

        ComputeContext::dispatch("fqms_copy_if_count", 1, &[
            self.cif_flags.wgpu_buffer(),
            self.cif_offsets.wgpu_buffer(),
            self.cif_count.wgpu_buffer(),
        ], Some(&p_n));

        self.cif_temp.copy_to(&mut self.triangles.idx);

        self.read_i32(&self.cif_count) as usize
    }

    // ===================================================================
    // Download result
    // ===================================================================

    pub fn download_result(&self) -> (Mat, Mat) {
        let mut verts = Mat::new("output_vertices");
        ComputeContext::memcpy_to_host_partial(
            &self.vertices.o_p, &mut verts, self.new_num_vertices, 1);

        let mut faces = Mat::new("output_faces");
        ComputeContext::memcpy_to_host_partial(
            &self.triangles.o_v, &mut faces, self.new_num_faces * 3, 1);

        (verts, faces)
    }

    // ===================================================================
    // Helpers
    // ===================================================================

    fn read_i32(&self, buf: &GpuMatrix) -> i32 {
        let mut m = Mat::new("tmp");
        buf.download(&mut m);
        m.as_slice::<i32>()[0]
    }
}
