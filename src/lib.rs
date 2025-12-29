//! nalgebra linear algebra extension for Nostos.
//!
//! Provides dynamic vector and matrix operations using the nalgebra library.
//! Unlike glam (fixed-size Vec2-4), nalgebra supports arbitrary-length vectors.

use nostos_extension::*;
use nalgebra::{DVector, DMatrix};

declare_extension!("nalgebra", "0.1.0", register);

fn register(reg: &mut ExtRegistry) {
    // DVector operations (dynamic-sized vectors)
    reg.add("Nalgebra.dvec", dvec_new);
    reg.add("Nalgebra.dvecZeros", dvec_zeros);
    reg.add("Nalgebra.dvecOnes", dvec_ones);
    reg.add("Nalgebra.dvecAdd", dvec_add);
    reg.add("Nalgebra.dvecSub", dvec_sub);
    reg.add("Nalgebra.dvecScale", dvec_scale);
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
}

// ==================== Helper Functions ====================

// Convert DVector to Value (list of floats)
fn dvec_to_value(v: &DVector<f64>) -> Value {
    Value::List(std::sync::Arc::new(
        v.iter().map(|&x| Value::Float(x)).collect()
    ))
}

// Convert Value (list) to DVector
fn value_to_dvec(v: &Value) -> Result<DVector<f64>, String> {
    let list = v.as_list()?;
    let data: Result<Vec<f64>, _> = list.iter().map(|x| x.as_f64()).collect();
    Ok(DVector::from_vec(data?))
}

// Convert DMatrix to Value (nested list, row-major for readability)
fn dmat_to_value(m: &DMatrix<f64>) -> Value {
    let rows: Vec<Value> = (0..m.nrows())
        .map(|i| {
            Value::List(std::sync::Arc::new(
                (0..m.ncols())
                    .map(|j| Value::Float(m[(i, j)]))
                    .collect()
            ))
        })
        .collect();
    Value::List(std::sync::Arc::new(rows))
}

// Convert Value (nested list) to DMatrix
fn value_to_dmat(v: &Value) -> Result<DMatrix<f64>, String> {
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

    // nalgebra uses column-major, so we need to transpose
    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}

// ==================== DVector Operations ====================

fn dvec_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let list = args[0].as_list()?;
    let data: Result<Vec<f64>, _> = list.iter().map(|x| x.as_f64()).collect();
    Ok(dvec_to_value(&DVector::from_vec(data?)))
}

fn dvec_zeros(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    Ok(dvec_to_value(&DVector::zeros(n)))
}

fn dvec_ones(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    Ok(dvec_to_value(&DVector::from_element(n, 1.0)))
}

fn dvec_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dvec(&args[0])?;
    let b = value_to_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_to_value(&(a + b)))
}

fn dvec_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dvec(&args[0])?;
    let b = value_to_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_to_value(&(a - b)))
}

fn dvec_scale(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    let s = args[1].as_f64()?;
    Ok(dvec_to_value(&(v * s)))
}

fn dvec_dot(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dvec(&args[0])?;
    let b = value_to_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(Value::Float(a.dot(&b)))
}

fn dvec_norm(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    Ok(Value::Float(v.norm()))
}

fn dvec_normalize(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    let norm = v.norm();
    if norm == 0.0 {
        return Err("Cannot normalize zero vector".to_string());
    }
    Ok(dvec_to_value(&v.normalize()))
}

fn dvec_len(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    Ok(Value::Int(v.len() as i64))
}

fn dvec_get(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    let i = args[1].as_i64()? as usize;
    if i >= v.len() {
        return Err(format!("Index {} out of bounds for vector of length {}", i, v.len()));
    }
    Ok(Value::Float(v[i]))
}

fn dvec_set(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let mut v = value_to_dvec(&args[0])?;
    let i = args[1].as_i64()? as usize;
    let val = args[2].as_f64()?;
    if i >= v.len() {
        return Err(format!("Index {} out of bounds for vector of length {}", i, v.len()));
    }
    v[i] = val;
    Ok(dvec_to_value(&v))
}

fn dvec_component_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dvec(&args[0])?;
    let b = value_to_dvec(&args[1])?;
    if a.len() != b.len() {
        return Err(format!("Vector length mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(dvec_to_value(&a.component_mul(&b)))
}

fn dvec_sum(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    Ok(Value::Float(v.sum()))
}

fn dvec_min(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    if v.is_empty() {
        return Err("Cannot find min of empty vector".to_string());
    }
    Ok(Value::Float(v.min()))
}

fn dvec_max(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_dvec(&args[0])?;
    if v.is_empty() {
        return Err("Cannot find max of empty vector".to_string());
    }
    Ok(Value::Float(v.max()))
}

// ==================== DMatrix Operations ====================

fn dmat_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    value_to_dmat(&args[0]).map(|m| dmat_to_value(&m))
}

fn dmat_identity(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let n = args[0].as_i64()? as usize;
    Ok(dmat_to_value(&DMatrix::identity(n, n)))
}

fn dmat_zeros(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows = args[0].as_i64()? as usize;
    let cols = args[1].as_i64()? as usize;
    Ok(dmat_to_value(&DMatrix::zeros(rows, cols)))
}

fn dmat_ones(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows = args[0].as_i64()? as usize;
    let cols = args[1].as_i64()? as usize;
    Ok(dmat_to_value(&DMatrix::from_element(rows, cols, 1.0)))
}

fn dmat_from_rows(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let rows_list = args[0].as_list()?;
    if rows_list.is_empty() {
        return Err("Cannot create matrix from empty list of rows".to_string());
    }

    let mut rows: Vec<DVector<f64>> = Vec::new();
    for row in rows_list.iter() {
        rows.push(value_to_dvec(row)?);
    }

    let ncols = rows[0].len();
    for row in &rows {
        if row.len() != ncols {
            return Err("All rows must have the same length".to_string());
        }
    }

    let m = DMatrix::from_rows(&rows.iter().map(|r| r.transpose()).collect::<Vec<_>>());
    Ok(dmat_to_value(&m))
}

fn dmat_from_cols(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let cols_list = args[0].as_list()?;
    if cols_list.is_empty() {
        return Err("Cannot create matrix from empty list of columns".to_string());
    }

    let mut cols: Vec<DVector<f64>> = Vec::new();
    for col in cols_list.iter() {
        cols.push(value_to_dvec(col)?);
    }

    let nrows = cols[0].len();
    for col in &cols {
        if col.len() != nrows {
            return Err("All columns must have the same length".to_string());
        }
    }

    let m = DMatrix::from_columns(&cols);
    Ok(dmat_to_value(&m))
}

fn dmat_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dmat(&args[0])?;
    let b = value_to_dmat(&args[1])?;
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(format!("Matrix dimension mismatch: {}x{} vs {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()));
    }
    Ok(dmat_to_value(&(a + b)))
}

fn dmat_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dmat(&args[0])?;
    let b = value_to_dmat(&args[1])?;
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(format!("Matrix dimension mismatch: {}x{} vs {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()));
    }
    Ok(dmat_to_value(&(a - b)))
}

fn dmat_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_dmat(&args[0])?;
    let b = value_to_dmat(&args[1])?;
    if a.ncols() != b.nrows() {
        return Err(format!("Matrix multiplication dimension mismatch: {}x{} * {}x{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()));
    }
    Ok(dmat_to_value(&(a * b)))
}

fn dmat_mul_vec(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    let v = value_to_dvec(&args[1])?;
    if m.ncols() != v.len() {
        return Err(format!("Matrix-vector dimension mismatch: {}x{} * {}",
            m.nrows(), m.ncols(), v.len()));
    }
    Ok(dvec_to_value(&(m * v)))
}

fn dmat_scale(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    let s = args[1].as_f64()?;
    Ok(dmat_to_value(&(m * s)))
}

fn dmat_transpose(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    Ok(dmat_to_value(&m.transpose()))
}

fn dmat_rows(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    Ok(Value::Int(m.nrows() as i64))
}

fn dmat_cols(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    Ok(Value::Int(m.ncols() as i64))
}

fn dmat_get(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    let row = args[1].as_i64()? as usize;
    let col = args[2].as_i64()? as usize;
    if row >= m.nrows() || col >= m.ncols() {
        return Err(format!("Index ({}, {}) out of bounds for {}x{} matrix",
            row, col, m.nrows(), m.ncols()));
    }
    Ok(Value::Float(m[(row, col)]))
}

fn dmat_set(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let mut m = value_to_dmat(&args[0])?;
    let row = args[1].as_i64()? as usize;
    let col = args[2].as_i64()? as usize;
    let val = args[3].as_f64()?;
    if row >= m.nrows() || col >= m.ncols() {
        return Err(format!("Index ({}, {}) out of bounds for {}x{} matrix",
            row, col, m.nrows(), m.ncols()));
    }
    m[(row, col)] = val;
    Ok(dmat_to_value(&m))
}

fn dmat_get_row(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    let row = args[1].as_i64()? as usize;
    if row >= m.nrows() {
        return Err(format!("Row {} out of bounds for {}x{} matrix", row, m.nrows(), m.ncols()));
    }
    let row_vec: DVector<f64> = m.row(row).transpose();
    Ok(dvec_to_value(&row_vec))
}

fn dmat_get_col(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    let col = args[1].as_i64()? as usize;
    if col >= m.ncols() {
        return Err(format!("Column {} out of bounds for {}x{} matrix", col, m.nrows(), m.ncols()));
    }
    let col_vec: DVector<f64> = m.column(col).into();
    Ok(dvec_to_value(&col_vec))
}

fn dmat_trace(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    if m.nrows() != m.ncols() {
        return Err("Trace is only defined for square matrices".to_string());
    }
    Ok(Value::Float(m.trace()))
}

fn dmat_determinant(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    if m.nrows() != m.ncols() {
        return Err("Determinant is only defined for square matrices".to_string());
    }
    Ok(Value::Float(m.determinant()))
}

fn dmat_inverse(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_dmat(&args[0])?;
    if m.nrows() != m.ncols() {
        return Err("Inverse is only defined for square matrices".to_string());
    }
    match m.try_inverse() {
        Some(inv) => Ok(dmat_to_value(&inv)),
        None => Err("Matrix is singular (not invertible)".to_string()),
    }
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

        // Create a vector
        let v = dvec_new(&[Value::List(std::sync::Arc::new(vec![
            Value::Float(1.0), Value::Float(2.0), Value::Float(3.0),
            Value::Float(4.0), Value::Float(5.0)
        ]))], &ctx).unwrap();

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
    }

    #[test]
    fn test_dmat_operations() {
        let ctx = make_ctx();

        // Create 2x2 identity
        let identity = dmat_identity(&[Value::Int(2)], &ctx).unwrap();

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
