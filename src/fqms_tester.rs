use forge_cv::{ComputeContext, GpuMatrix, Mat, PrefixScan, ScalarType};
use gpu_fqms::Fqms;
use std::fs;

/// Load an ASCII PLY file. Returns (vertices as flat f32 xyz, faces as flat i32 with 4 ints per face).
fn load_ply(path: &str) -> (Vec<f32>, Vec<i32>, usize, usize) {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read '{}': {}", path, e));

    let mut lines = content.lines();
    let mut num_vertices = 0;
    let mut num_faces = 0;

    // Parse header
    for line in &mut lines {
        let line = line.trim();
        if line.starts_with("element vertex") {
            num_vertices = line.split_whitespace().last().unwrap().parse().unwrap();
        } else if line.starts_with("element face") {
            num_faces = line.split_whitespace().last().unwrap().parse().unwrap();
        } else if line == "end_header" {
            break;
        }
    }

    // Parse vertices (x, y, z, [ignore rest])
    let mut vertices = Vec::with_capacity(num_vertices * 3);
    for _ in 0..num_vertices {
        let line = lines.next().expect("Unexpected end of vertex data");
        let mut parts = line.split_whitespace();
        let x: f32 = parts.next().unwrap().parse().unwrap();
        let y: f32 = parts.next().unwrap().parse().unwrap();
        let z: f32 = parts.next().unwrap().parse().unwrap();
        vertices.push(x);
        vertices.push(y);
        vertices.push(z);
    }

    // Parse faces (count v0 v1 v2 ...) — add object_id=0
    let mut faces = Vec::with_capacity(num_faces * 4);
    for _ in 0..num_faces {
        let line = lines.next().expect("Unexpected end of face data");
        let mut parts = line.split_whitespace();
        let count: usize = parts.next().unwrap().parse().unwrap();
        assert_eq!(count, 3, "Only triangle meshes supported");
        let v0: i32 = parts.next().unwrap().parse().unwrap();
        let v1: i32 = parts.next().unwrap().parse().unwrap();
        let v2: i32 = parts.next().unwrap().parse().unwrap();
        faces.push(v0);
        faces.push(v1);
        faces.push(v2);
        faces.push(0); // object_id
    }

    (vertices, faces, num_vertices, num_faces)
}

fn test_prefix_scan() {
    println!("\n--- Prefix Scan Test ---");

    // Small test: [3, 1, 4, 1, 5] → exclusive scan → [0, 3, 4, 8, 9]
    let input_data: Vec<i32> = vec![3, 1, 4, 1, 5];
    let n = input_data.len();

    let input_mat = Mat::from_slice("scan_input", n, 1, ScalarType::Int32, 1, &input_data);
    let mut input_gpu = GpuMatrix::new();
    input_gpu.upload(&input_mat);

    let output_gpu = ComputeContext::malloc(n, 1, ScalarType::Int32, 1);

    let mut scan = PrefixScan::new();
    scan.exclusive_scan(&input_gpu, &output_gpu);

    let mut result = Mat::new("scan_result");
    output_gpu.download(&mut result);
    let result_data: &[i32] = result.as_slice();

    let expected: Vec<i32> = vec![0, 3, 4, 8, 9];
    println!("Input:    {:?}", input_data);
    println!("Expected: {:?}", expected);
    println!("Got:      {:?}", &result_data[..n]);

    assert_eq!(&result_data[..n], &expected[..], "Prefix scan mismatch!");
    println!("PASSED!");
}

fn main() {
    ComputeContext::init();
    println!("Device: {}", ComputeContext::device_name());

    // Test prefix scan
    test_prefix_scan();

    // Load and upload mesh
    let (vertices, faces, num_v, num_f) = load_ply("bunny.ply");
    println!("\nLoaded bunny.ply: {} vertices, {} faces", num_v, num_f);

    let vert_mat = Mat::from_slice("vertices", num_v, 1, ScalarType::Float32, 3, &vertices);
    let face_mat = Mat::from_slice("faces", num_f, 1, ScalarType::Int32, 4, &faces);

    let mut fqms = Fqms::new();
    fqms.upload(&face_mat, &vert_mat);

    println!("Uploaded: {} vertices, {} faces", fqms.num_vertices(), fqms.num_faces());

    // Verify init_mesh_data results
    println!("\n--- Verifying init_mesh_data ---");

    // Check tcount: should sum to num_faces * 3 (each face contributes 3 half-edges)
    let mut tcount_mat = Mat::new("tcount");
    fqms.vertices.tcount.download(&mut tcount_mat);
    let tcount: &[i32] = tcount_mat.as_slice();
    let total_he: i64 = tcount.iter().map(|&x| x as i64).sum();
    println!("tcount sum: {} (expected {})", total_he, num_f * 3);
    assert_eq!(total_he, (num_f * 3) as i64, "tcount sum mismatch");

    // Check tstart: should be exclusive scan of tcount
    let mut tstart_mat = Mat::new("tstart");
    fqms.vertices.tstart.download(&mut tstart_mat);
    let tstart: &[i32] = tstart_mat.as_slice();
    println!("tstart[0]={}, tstart[1]={}, last={}", tstart[0], tstart[1], tstart[num_v - 1]);
    assert_eq!(tstart[0], 0, "tstart[0] should be 0");

    // Check border: at least some vertices should be marked
    let mut border_mat = Mat::new("border");
    fqms.vertices.border.download(&mut border_mat);
    let border: &[i32] = border_mat.as_slice();
    let num_border: usize = border.iter().filter(|&&b| b != 0).count();
    println!("Border vertices: {} / {}", num_border, num_v);

    // Check err: no NaN in first few faces
    let mut err_mat = Mat::new("err");
    fqms.triangles.err.download(&mut err_mat);
    let err: &[f32] = err_mat.as_slice();
    let mut nan_count = 0;
    for i in 0..num_f * 4 {
        if err[i].is_nan() { nan_count += 1; }
    }
    println!("Error NaN count: {} / {}", nan_count, num_f * 4);
    println!("err[0..4] = [{:.6}, {:.6}, {:.6}, {:.6}]", err[0], err[1], err[2], err[3]);

    // Check idx: should be 0,1,2,...,n-1
    let mut idx_mat = Mat::new("idx");
    fqms.triangles.idx.download(&mut idx_mat);
    let idx: &[i32] = idx_mat.as_slice();
    assert_eq!(idx[0], 0);
    assert_eq!(idx[num_f - 1], (num_f - 1) as i32);
    println!("idx: OK (0..{})", num_f - 1);

    println!("--- All checks passed! ---");

    // ===================================================================
    // Simplify: 69k faces → 10k faces
    // ===================================================================
    let target = 10000;
    println!("\n--- Simplifying to {} faces ---", target);
    let start = std::time::Instant::now();
    fqms.simplify(target);
    let elapsed = start.elapsed();
    println!("Done in {:.1?}", elapsed);
    println!("Result: {} vertices, {} faces",
        fqms.result_num_vertices(), fqms.result_num_faces());

    // Download and basic sanity check
    let (result_verts, result_faces) = fqms.download_result();
    let rv: &[f32] = result_verts.as_slice();
    let rf: &[i32] = result_faces.as_slice();
    println!("Downloaded: {} vertex floats, {} face indices",
        rv.len(), rf.len());

    // Verify face indices are in range
    let nv_out = fqms.result_num_vertices() as i32;
    let mut bad = 0;
    for &idx in rf.iter() {
        if idx < 0 || idx >= nv_out { bad += 1; }
    }
    println!("Out-of-range vertex indices: {}", bad);
    assert_eq!(bad, 0, "Some face indices are out of range!");
    println!("--- Simplification OK! ---");

    // Save simplified mesh as PLY
    let out_path = "bunny_simplified.ply";
    save_ply(out_path, rv, rf, fqms.result_num_vertices(), fqms.result_num_faces());
    println!("\nSaved to {}", out_path);
}

fn save_ply(path: &str, vertices: &[f32], faces: &[i32], num_v: usize, num_f: usize) {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    writeln!(f, "ply").unwrap();
    writeln!(f, "format ascii 1.0").unwrap();
    writeln!(f, "element vertex {}", num_v).unwrap();
    writeln!(f, "property float x").unwrap();
    writeln!(f, "property float y").unwrap();
    writeln!(f, "property float z").unwrap();
    writeln!(f, "element face {}", num_f).unwrap();
    writeln!(f, "property list uchar int vertex_indices").unwrap();
    writeln!(f, "end_header").unwrap();
    for i in 0..num_v {
        let b = i * 3;
        writeln!(f, "{} {} {}", vertices[b], vertices[b + 1], vertices[b + 2]).unwrap();
    }
    for i in 0..num_f {
        let b = i * 3;
        writeln!(f, "3 {} {} {}", faces[b], faces[b + 1], faces[b + 2]).unwrap();
    }
}
