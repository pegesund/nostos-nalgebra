//! nalgebra linear algebra extension for Nostos.
//!
//! Provides dynamic vector and matrix operations using the nalgebra library.
//! Uses GC-managed native handles to store data directly, avoiding copies.

use nostos_extension::*;
use nalgebra::{DVector, DMatrix};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Global counters to track allocations and cleanups (for testing)
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static CLEANUP_COUNT: AtomicU64 = AtomicU64::new(0);

declare_extension!("nalgebra", "0.1.0", register);

// Type IDs for native handles
const TYPE_DVECTOR: u64 = 1;
const TYPE_DMATRIX: u64 = 2;

// Cleanup function called by GC when a handle is collected
fn nalgebra_cleanup(ptr: usize, type_id: u64) {
    CLEANUP_COUNT.fetch_add(1, Ordering::Relaxed);
    match type_id {
        TYPE_DVECTOR => {
            // Reconstruct and drop the Box<DVector<f64>>
            unsafe {
                let _ = Box::from_raw(ptr as *mut DVector<f64>);
            }
        }
        TYPE_DMATRIX => {
            // Reconstruct and drop the Box<DMatrix<f64>>
            unsafe {
                let _ = Box::from_raw(ptr as *mut DMatrix<f64>);
            }
        }
        _ => {
            eprintln!("nalgebra_cleanup: unknown type_id {}", type_id);
        }
    }
}

fn register(reg: &mut ExtRegistry) {
    // DVector operations (dynamic-sized vectors)
    reg.add("Nalgebra.dvec", dvec_new);
    reg.add("Nalgebra.dvecZeros", dvec_zeros);
    reg.add("Nalgebra.dvecOnes", dvec_ones);
    reg.add("Nalgebra.dvecAdd", dvec_add);
    reg.add("Nalgebra.dvecSub", dvec_sub);
    reg.add("Nalgebra.dvecScale", dvec_scale);
    reg.add("Nalgebra.dvecAddScalar", dvec_add_scalar);
    reg.add("Nalgebra.dvecSubScalar", dvec_sub_scalar);
    reg.add("Nalgebra.dvecDivScalar", dvec_div_scalar);
    reg.add("Nalgebra.dvecDot", dvec_dot);
    reg.add("Nalgebra.dvecNorm", dvec_norm);
    reg.add("Nalgebra.dvecNormalize", dvec_normalize);
    reg.add("Nalgebra.dvecLen", dvec_len);
    reg.add("Nalgebra.dvecGet", dvec_get);
    reg.add("Nalgebra.dvecSet", dvec_set);
    reg.add("Nalgebra.dvecMap", dvec_component_mul);
    reg.add("Nalgebra.dvecSum", dvec_sum);
    reg.add("Nalgebra.dvecMin", dvec_min);
    reg.add("Nalgebra.dvecMax", dvec_max);
    reg.add("Nalgebra.dvecDiv", dvec_div);
    reg.add("Nalgebra.dvecToList", dvec_to_list);
    reg.add("Nalgebra.dvecRange", dvec_range);
    reg.add("Nalgebra.dvecLinspace", dvec_linspace);
    reg.add("Nalgebra.dvecFromSeed", dvec_from_seed);
    reg.add("Nalgebra.dvecMake", dvec_make);

    // DMatrix operations (dynamic-sized matrices)
    reg.add("Nalgebra.dmat", dmat_new);
    reg.add("Nalgebra.dmatIdentity", dmat_identity);
    reg.add("Nalgebra.dmatZeros", dmat_zeros);
    reg.add("Nalgebra.dmatOnes", dmat_ones);
    reg.add("Nalgebra.dmatFromRows", dmat_from_rows);
    reg.add("Nalgebra.dmatFromCols", dmat_from_cols);
    reg.add("Nalgebra.dmatAdd", dmat_add);
    reg.add("Nalgebra.dmatSub", dmat_sub);
    reg.add("Nalgebra.dmatMul", dmat_mul);
    reg.add("Nalgebra.dmatMulVec", dmat_mul_vec);
    reg.add("Nalgebra.dmatScale", dmat_scale);
    reg.add("Nalgebra.dmatTranspose", dmat_transpose);
    reg.add("Nalgebra.dmatRows", dmat_rows);
    reg.add("Nalgebra.dmatCols", dmat_cols);
    reg.add("Nalgebra.dmatGet", dmat_get);
    reg.add("Nalgebra.dmatSet", dmat_set);
    reg.add("Nalgebra.dmatGetRow", dmat_get_row);
    reg.add("Nalgebra.dmatGetCol", dmat_get_col);
    reg.add("Nalgebra.dmatTrace", dmat_trace);
    reg.add("Nalgebra.dmatDeterminant", dmat_determinant);
    reg.add("Nalgebra.dmatInverse", dmat_inverse);
    reg.add("Nalgebra.dmatDiag", dmat_diag);
    reg.add("Nalgebra.dmatPow", dmat_pow);
    reg.add("Nalgebra.dmatToList", dmat_to_list);
    reg.add("Nalgebra.dmatFromSeed", dmat_from_seed);

    // Stats functions for GC testing
    reg.add("Nalgebra.allocCount", stats_alloc_count);
    reg.add("Nalgebra.cleanupCount", stats_cleanup_count);
    reg.add("Nalgebra.resetStats", stats_reset);
}

// ==================== Helper Functions ====================

// Create a GcHandle for a DVector
fn dvec_handle(v: DVector<f64>) -> Value {
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    Value::gc_handle(Box::new(v), TYPE_DVECTOR, nalgebra_cleanup)
}

// Create a GcHandle for a DMatrix
fn dmat_handle(m: DMatrix<f64>) -> Value {
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    Value::gc_handle(Box::new(m), TYPE_DMATRIX, nalgebra_cleanup)
}

// Extract DVector from a GcHandle or convert from List (for backward compatibility)
fn get_dvec(v: &Value) -> Result<&DVector<f64>, String> {
    match v {
        Value::GcHandle(h) => {
            if h.type_id != TYPE_DVECTOR {
                return Err(format!("Expected DVector handle (type={}), got type={}", TYPE_DVECTOR, h.type_id));
            }
            if h.ptr == 0 {
                return Err("Invalid DVector handle (null pointer)".to_string());
            }
            Ok(unsafe { &*(h.ptr as *const DVector<f64>) })
        }
        Value::List(_) => {
            Err("Expected native handle, got List. Use dvec() to create native vectors.".to_string())
        }
        _ => Err(format!("Expected DVector handle, got {:?}", v.type_name())),
    }
}

// Extract DMatrix from a GcHandle
fn get_dmat(v: &Value) -> Result<&DMatrix<f64>, String> {
    match v {
        Value::GcHandle(h) => {
            if h.type_id != TYPE_DMATRIX {
                return Err(format!("Expected DMatrix handle (type={}), got type={}", TYPE_DMATRIX, h.type_id));
            }
            if h.ptr == 0 {
                return Err("Invalid DMatrix handle (null pointer)".to_string());
            }
            Ok(unsafe { &*(h.ptr as *const DMatrix<f64>) })
        }
        _ => Err(format!("Expected DMatrix handle, got {:?}", v.type_name())),
    }
}

// Convert list of floats to DVector
fn list_to_dvec(v: &Value) -> Result<DVector<f64>, String> {
    let list = v.as_list()?;
    let data: Result<Vec<f64>, _> = list.iter().map(|x| x.as_f64()).collect();
    Ok(DVector::from_vec(data?))
}

// Convert nested list to DMatrix
fn list_to_dmat(v: &Value) -> Result<DMatrix<f64>, String> {
    let rows = v.as_list()?;
    if rows.is_empty() {
        return Err("Matrix cannot be empty".to_string());
    }

    let nrows = rows.len();
    let first_row = rows[0].as_list()?;
    let ncols = first_row.len();

    let mut data = Vec::with_capacity(nrows * ncols);
    for row in rows.iter() {
        let row_vals = row.as_list()?;
        if row_vals.len() != ncols {
            return Err("All rows must have the same length".to_string());
        }
        for val in row_vals.iter() {
            data.push(val.as_f64()?);
        }
    }

    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}

// ==================== DVector Operations ====================

fn dvec_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = list_to_dvec(&args[0])?;
    Ok(dvec_handle(v))
}

fn dvec_zeros(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    Ok(dvec_handle(DVector::zeros(n)))
}

fn dvec_ones(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    Ok(dvec_handle(DVector::from_element(n, 1.0)))
}

fn dvec_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dvec(&args[0])?;
    let b = get_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_handle(a + b))
}

fn dvec_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dvec(&args[0])?;
    let b = get_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_handle(a - b))
}

fn dvec_scale(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let s = args[1].as_f64()?;
    Ok(dvec_handle(v * s))
}

fn dvec_add_scalar(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let s = args[1].as_f64()?;
    Ok(dvec_handle(v.add_scalar(s)))
}

fn dvec_sub_scalar(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let s = args[1].as_f64()?;
    Ok(dvec_handle(v.add_scalar(-s)))
}

fn dvec_div_scalar(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let s = args[1].as_f64()?;
    if s == 0.0 {
        return Err("Division by zero".to_string());
    }
    Ok(dvec_handle(v / s))
}

fn dvec_dot(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dvec(&args[0])?;
    let b = get_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(Value::Float(a.dot(b)))
}

fn dvec_norm(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    Ok(Value::Float(v.norm()))
}

fn dvec_normalize(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let norm = v.norm();
    if norm == 0.0 {
        return Err("Cannot normalize zero vector".to_string());
    }
    Ok(dvec_handle(v.normalize()))
}

fn dvec_len(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    Ok(Value::Int(v.len() as i64))
}

fn dvec_get(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let i = args[1].as_i64()? as usize;
    if i >= v.len() {
        return Err(format!("Index {} out of bounds for vector of length {}", i, v.len()));
    }
    Ok(Value::Float(v[i]))
}

fn dvec_set(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let i = args[1].as_i64()? as usize;
    let val = args[2].as_f64()?;
    if i >= v.len() {
        return Err(format!("Index {} out of bounds for vector of length {}", i, v.len()));
    }
    let mut new_v = v.clone();
    new_v[i] = val;
    Ok(dvec_handle(new_v))
}

fn dvec_component_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dvec(&args[0])?;
    let b = get_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_handle(a.component_mul(b)))
}

fn dvec_sum(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    Ok(Value::Float(v.sum()))
}

fn dvec_min(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    if v.is_empty() {
        return Err("Cannot find min of empty vector".to_string());
    }
    Ok(Value::Float(v.min()))
}

fn dvec_max(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    if v.is_empty() {
        return Err("Cannot find max of empty vector".to_string());
    }
    Ok(Value::Float(v.max()))
}

fn dvec_div(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dvec(&args[0])?;
    let b = get_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_handle(a.component_div(b)))
}

// Convert native handle back to list (for interop)
fn dvec_to_list(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    Ok(Value::List(Arc::new(
        v.iter().map(|&x| Value::Float(x)).collect()
    )))
}

// Create vector from integer range [start, end) as floats
fn dvec_range(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let start = args[0].as_i64()?;
    let end = args[1].as_i64()?;
    let data: Vec<f64> = (start..end).map(|x| x as f64).collect();
    Ok(dvec_handle(DVector::from_vec(data)))
}

// Create vector with n evenly spaced values from start to end
fn dvec_linspace(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let start = args[0].as_f64()?;
    let end = args[1].as_f64()?;
    let n = args[2].as_i64()? as usize;
    if n < 2 {
        return Err("linspace requires at least 2 points".to_string());
    }
    let step = (end - start) / (n - 1) as f64;
    let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    Ok(dvec_handle(DVector::from_vec(data)))
}

// Create vector of length n filled with initial value
fn dvec_make(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    let value = args[1].as_f64()?;
    Ok(dvec_handle(DVector::from_element(n, value)))
}

// Create pseudo-random vector matching genList(n, seed) for benchmarking
fn dvec_from_seed(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    let seed = args[1].as_i64()?;
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = ((i as i64) * 1103515245 + seed) % 2147483647;
            ((x % 1000000) as f64) / 1000000.0
        })
        .collect();
    Ok(dvec_handle(DVector::from_vec(data)))
}

// ==================== DMatrix Operations ====================

fn dmat_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = list_to_dmat(&args[0])?;
    Ok(dmat_handle(m))
}

fn dmat_identity(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    Ok(dmat_handle(DMatrix::identity(n, n)))
}

fn dmat_zeros(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows = args[0].as_i64()? as usize;
    let cols = args[1].as_i64()? as usize;
    Ok(dmat_handle(DMatrix::zeros(rows, cols)))
}

fn dmat_ones(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows = args[0].as_i64()? as usize;
    let cols = args[1].as_i64()? as usize;
    Ok(dmat_handle(DMatrix::from_element(rows, cols, 1.0)))
}

fn dmat_from_rows(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows_list = args[0].as_list()?;
    if rows_list.is_empty() {
        return Err("Cannot create matrix from empty list of rows".to_string());
    }

    let mut rows: Vec<DVector<f64>> = Vec::new();
    for row in rows_list.iter() {
        // Handle both native handles and lists
        match row {
            Value::GcHandle(h) if h.type_id == TYPE_DVECTOR => {
                rows.push(get_dvec(row)?.clone());
            }
            Value::List(_) => {
                rows.push(list_to_dvec(row)?);
            }
            _ => return Err("Row must be a vector or list".to_string()),
        }
    }

    let ncols = rows[0].len();
    for row in &rows {
        if row.len() != ncols {
            return Err("All rows must have the same length".to_string());
        }
    }

    let m = DMatrix::from_rows(&rows.iter().map(|r| r.transpose()).collect::<Vec<_>>());
    Ok(dmat_handle(m))
}

fn dmat_from_cols(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let cols_list = args[0].as_list()?;
    if cols_list.is_empty() {
        return Err("Cannot create matrix from empty list of columns".to_string());
    }

    let mut cols: Vec<DVector<f64>> = Vec::new();
    for col in cols_list.iter() {
        match col {
            Value::GcHandle(h) if h.type_id == TYPE_DVECTOR => {
                cols.push(get_dvec(col)?.clone());
            }
            Value::List(_) => {
                cols.push(list_to_dvec(col)?);
            }
            _ => return Err("Column must be a vector or list".to_string()),
        }
    }

    let nrows = cols[0].len();
    for col in &cols {
        if col.len() != nrows {
            return Err("All columns must have the same length".to_string());
        }
    }

    let m = DMatrix::from_columns(&cols);
    Ok(dmat_handle(m))
}

fn dmat_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dmat(&args[0])?;
    let b = get_dmat(&args[1])?;
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(format!("Matrix dimension mismatch: {}x{} vs {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()));
    }
    Ok(dmat_handle(a + b))
}

fn dmat_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dmat(&args[0])?;
    let b = get_dmat(&args[1])?;
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(format!("Matrix dimension mismatch: {}x{} vs {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()));
    }
    Ok(dmat_handle(a - b))
}

fn dmat_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = get_dmat(&args[0])?;
    let b = get_dmat(&args[1])?;
    if a.ncols() != b.nrows() {
        return Err(format!("Matrix multiplication dimension mismatch: {}x{} * {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()));
    }
    Ok(dmat_handle(a * b))
}

fn dmat_mul_vec(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let v = get_dvec(&args[1])?;
    if m.ncols() != v.len() {
        return Err(format!("Matrix-vector dimension mismatch: {}x{} * {}",
            m.nrows(), m.ncols(), v.len()));
    }
    Ok(dvec_handle(m * v))
}

fn dmat_scale(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let s = args[1].as_f64()?;
    Ok(dmat_handle(m * s))
}

fn dmat_transpose(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    Ok(dmat_handle(m.transpose()))
}

fn dmat_rows(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    Ok(Value::Int(m.nrows() as i64))
}

fn dmat_cols(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    Ok(Value::Int(m.ncols() as i64))
}

fn dmat_get(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let row = args[1].as_i64()? as usize;
    let col = args[2].as_i64()? as usize;
    if row >= m.nrows() || col >= m.ncols() {
        return Err(format!("Index ({}, {}) out of bounds for {}x{} matrix",
            row, col, m.nrows(), m.ncols()));
    }
    Ok(Value::Float(m[(row, col)]))
}

fn dmat_set(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let row = args[1].as_i64()? as usize;
    let col = args[2].as_i64()? as usize;
    let val = args[3].as_f64()?;
    if row >= m.nrows() || col >= m.ncols() {
        return Err(format!("Index ({}, {}) out of bounds for {}x{} matrix",
            row, col, m.nrows(), m.ncols()));
    }
    let mut new_m = m.clone();
    new_m[(row, col)] = val;
    Ok(dmat_handle(new_m))
}

fn dmat_get_row(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let row = args[1].as_i64()? as usize;
    if row >= m.nrows() {
        return Err(format!("Row {} out of bounds for {}x{} matrix", row, m.nrows(), m.ncols()));
    }
    let row_vec: DVector<f64> = m.row(row).transpose();
    Ok(dvec_handle(row_vec))
}

fn dmat_get_col(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let col = args[1].as_i64()? as usize;
    if col >= m.ncols() {
        return Err(format!("Column {} out of bounds for {}x{} matrix", col, m.nrows(), m.ncols()));
    }
    let col_vec: DVector<f64> = m.column(col).into();
    Ok(dvec_handle(col_vec))
}

fn dmat_trace(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    if m.nrows() != m.ncols() {
        return Err("Trace is only defined for square matrices".to_string());
    }
    Ok(Value::Float(m.trace()))
}

fn dmat_determinant(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    if m.nrows() != m.ncols() {
        return Err("Determinant is only defined for square matrices".to_string());
    }
    Ok(Value::Float(m.determinant()))
}

fn dmat_inverse(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    if m.nrows() != m.ncols() {
        return Err("Inverse is only defined for square matrices".to_string());
    }
    match m.clone().try_inverse() {
        Some(inv) => Ok(dmat_handle(inv)),
        None => Err("Matrix is singular (not invertible)".to_string()),
    }
}

fn dmat_diag(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = get_dvec(&args[0])?;
    let n = v.len();
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = v[i];
    }
    Ok(dmat_handle(m))
}

fn dmat_pow(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let n = args[1].as_i64()?;
    if m.nrows() != m.ncols() {
        return Err("Matrix power is only defined for square matrices".to_string());
    }
    if n < 0 {
        return Err("Negative matrix powers not supported".to_string());
    }
    if n == 0 {
        return Ok(dmat_handle(DMatrix::identity(m.nrows(), m.ncols())));
    }
    let mut result = m.clone();
    for _ in 1..n {
        result = &result * m;
    }
    Ok(dmat_handle(result))
}

// Convert native handle back to nested list (for interop)
fn dmat_to_list(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = get_dmat(&args[0])?;
    let rows: Vec<Value> = (0..m.nrows())
        .map(|i| {
            Value::List(Arc::new(
                (0..m.ncols())
                    .map(|j| Value::Float(m[(i, j)]))
                    .collect()
            ))
        })
        .collect();
    Ok(Value::List(Arc::new(rows)))
}

// Create pseudo-random matrix matching genMatrixData(rows, cols, seed)
fn dmat_from_seed(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows = args[0].as_i64()? as usize;
    let cols = args[1].as_i64()? as usize;
    let seed = args[2].as_i64()?;

    let data: Vec<f64> = (0..rows)
        .flat_map(|r| {
            let row_seed = seed + (r as i64) * 1000;
            (0..cols).map(move |c| {
                let x = ((c as i64) * 1103515245 + row_seed) % 2147483647;
                ((x % 1000000) as f64) / 1000000.0
            })
        })
        .collect();
    Ok(dmat_handle(DMatrix::from_row_slice(rows, cols, &data)))
}

// ==================== Stats Functions (for GC testing) ====================

fn stats_alloc_count(_args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    Ok(Value::Int(ALLOC_COUNT.load(Ordering::Relaxed) as i64))
}

fn stats_cleanup_count(_args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    Ok(Value::Int(CLEANUP_COUNT.load(Ordering::Relaxed) as i64))
}

fn stats_reset(_args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    CLEANUP_COUNT.store(0, Ordering::Relaxed);
    Ok(Value::Unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> ExtContext {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ExtContext::new(rt.handle().clone(), tx, Pid(1))
    }

    #[test]
    fn test_dvec_operations() {
        let ctx = make_ctx();

        // Create a vector from list
        let v = dvec_new(&[Value::List(Arc::new(vec![
            Value::Float(1.0), Value::Float(2.0), Value::Float(3.0),
            Value::Float(4.0), Value::Float(5.0)
        ]))], &ctx).unwrap();

        // Verify it's a GcHandle
        assert!(matches!(v, Value::GcHandle(_)));

        // Check length
        let len = dvec_len(&[v.clone()], &ctx).unwrap();
        assert_eq!(len.as_i64().unwrap(), 5);

        // Check norm
        let norm = dvec_norm(&[v.clone()], &ctx).unwrap();
        let expected = (1.0_f64 + 4.0 + 9.0 + 16.0 + 25.0).sqrt();
        assert!((norm.as_f64().unwrap() - expected).abs() < 1e-10);

        // Check sum
        let sum = dvec_sum(&[v.clone()], &ctx).unwrap();
        assert_eq!(sum.as_f64().unwrap(), 15.0);

        // Add vectors
        let v2 = dvec_new(&[Value::List(Arc::new(vec![
            Value::Float(5.0), Value::Float(4.0), Value::Float(3.0),
            Value::Float(2.0), Value::Float(1.0)
        ]))], &ctx).unwrap();

        let sum_vec = dvec_add(&[v.clone(), v2.clone()], &ctx).unwrap();
        let sum_list = dvec_to_list(&[sum_vec], &ctx).unwrap();
        let sum_list = sum_list.as_list().unwrap();
        assert_eq!(sum_list.len(), 5);
        assert_eq!(sum_list[0].as_f64().unwrap(), 6.0);
    }

    #[test]
    fn test_dmat_operations() {
        let ctx = make_ctx();

        // Create 2x2 identity
        let identity = dmat_identity(&[Value::Int(2)], &ctx).unwrap();

        // Verify it's a GcHandle
        assert!(matches!(identity, Value::GcHandle(_)));

        // Check dimensions
        let rows = dmat_rows(&[identity.clone()], &ctx).unwrap();
        let cols = dmat_cols(&[identity.clone()], &ctx).unwrap();
        assert_eq!(rows.as_i64().unwrap(), 2);
        assert_eq!(cols.as_i64().unwrap(), 2);

        // Check trace (should be 2 for 2x2 identity)
        let trace = dmat_trace(&[identity.clone()], &ctx).unwrap();
        assert_eq!(trace.as_f64().unwrap(), 2.0);

        // Check determinant (should be 1 for identity)
        let det = dmat_determinant(&[identity.clone()], &ctx).unwrap();
        assert_eq!(det.as_f64().unwrap(), 1.0);
    }
}
