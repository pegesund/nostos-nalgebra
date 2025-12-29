# nostos-nalgebra

A Nostos extension providing dynamic-sized linear algebra operations using the [nalgebra](https://nalgebra.org/) library.

Unlike `glam` (which provides fixed-size Vec2-4), `nalgebra` supports arbitrary-length vectors and matrices, making it suitable for machine learning, scientific computing, and general linear algebra.

## Features

- **Operator overloading**: Use `+`, `-`, `*`, `/` directly on vectors and matrices
- **Native handles**: Data is stored directly in nalgebra structures, not copied to/from Nostos lists
- **GC integration**: Memory is automatically freed when handles go out of scope
- **~9x faster** than pure Nostos list operations for vector/matrix math
- **SIMD acceleration**: Leverages CPU vector instructions via [simba](https://docs.rs/simba/)

## Installation

Add to your `nostos.toml`:

```toml
[extensions]
nalgebra = { git = "https://github.com/pegesund/nostos-nalgebra" }
```

## Usage

```nostos
import nalgebra

main() = {
    # Create vectors with the vec() constructor
    v1 = nalgebra.vec([1.0, 2.0, 3.0, 4.0, 5.0])
    v2 = nalgebra.vec([5.0, 4.0, 3.0, 2.0, 1.0])

    # Operator overloading!
    sum = v1 + v2
    diff = v1 - v2
    prod = v1 * v2  # Element-wise multiplication

    # Vector operations
    dot = nalgebra.vecDot(v1, v2)
    norm = nalgebra.vecNorm(v1)
    scaled = nalgebra.vecScale(v1, 2.0)

    println("Dot product: ${dot}")
    println("Norm: ${norm}")

    # Create matrices
    m1 = nalgebra.mat([[1.0, 2.0], [3.0, 4.0]])
    m2 = nalgebra.mat([[5.0, 6.0], [7.0, 8.0]])

    # Matrix operator overloading
    mSum = m1 + m2
    mProd = m1 * m2  # Matrix multiplication

    # Matrix operations
    det = nalgebra.matDeterminant(m1)
    inv = nalgebra.matInverse(m1)
    identity = nalgebra.matIdentity(3)

    println("Determinant: ${det}")

    0
}
```

## API Reference

### Vec (Dynamic Vectors)

#### Operators
Vectors support `+`, `-`, `*`, `/` operators directly:
```nostos
v1 + v2   # Vector addition
v1 - v2   # Vector subtraction
v1 * v2   # Element-wise multiplication
v1 / v2   # Element-wise division
```

#### Constructors
| Function | Description |
|----------|-------------|
| `vec(list)` | Create vector from list of floats |
| `vecZeros(n)` | Create zero vector of length n |
| `vecOnes(n)` | Create vector of ones of length n |

#### Operations
| Function | Description |
|----------|-------------|
| `vecDot(a, b)` | Dot product |
| `vecNorm(v)` | Euclidean norm (length) |
| `vecNormalize(v)` | Normalize to unit length |
| `vecLen(v)` | Get number of elements |
| `vecGet(v, i)` | Get element at index i |
| `vecSum(v)` | Sum of all elements |
| `vecMin(v)` | Minimum element |
| `vecMax(v)` | Maximum element |
| `vecScale(v, s)` | Multiply vector by scalar |
| `vecDistance(a, b)` | Euclidean distance between vectors |

### Mat (Dynamic Matrices)

#### Operators
Matrices support `+`, `-`, `*`, `/` operators directly:
```nostos
m1 + m2   # Matrix addition
m1 - m2   # Matrix subtraction
m1 * m2   # Matrix multiplication
m1 / m2   # m1 * inverse(m2)
```

#### Constructors
| Function | Description |
|----------|-------------|
| `mat(rows)` | Create matrix from nested list |
| `matIdentity(n)` | Create n x n identity matrix |
| `matZeros(rows, cols)` | Create zero matrix |
| `matOnes(rows, cols)` | Create matrix of ones |

#### Operations
| Function | Description |
|----------|-------------|
| `matRows(m)` | Get number of rows |
| `matCols(m)` | Get number of columns |
| `matGet(m, row, col)` | Get element |
| `matTranspose(m)` | Transpose matrix |
| `matTrace(m)` | Trace (sum of diagonal) |
| `matDeterminant(m)` | Determinant |
| `matInverse(m)` | Matrix inverse |
| `matScale(m, s)` | Multiply matrix by scalar |
| `matPow(m, n)` | Matrix power |
| `matMulVec(m, v)` | Matrix-vector multiplication |
| `matGetRow(m, row)` | Get row as vector |
| `matGetCol(m, col)` | Get column as vector |

### Raw Native Handle Functions

For lower-level access or when you need direct native handles without the wrapper types, use the `dvec*` and `dmat*` functions:

```nostos
# Direct native handle API
v = nalgebra.dvec([1.0, 2.0, 3.0])
result = nalgebra.dvecAdd(v, nalgebra.dvecOnes(3))
```

| Vector Functions | Matrix Functions |
|-----------------|------------------|
| `dvec`, `dvecZeros`, `dvecOnes` | `dmat`, `dmatIdentity`, `dmatZeros`, `dmatOnes` |
| `dvecAdd`, `dvecSub`, `dvecScale` | `dmatAdd`, `dmatSub`, `dmatScale` |
| `dvecDot`, `dvecNorm`, `dvecNormalize` | `dmatMul`, `dmatMulVec`, `dmatTranspose` |
| `dvecLen`, `dvecGet`, `dvecSet` | `dmatRows`, `dmatCols`, `dmatGet`, `dmatSet` |
| `dvecComponentMul`, `dvecSum` | `dmatTrace`, `dmatDeterminant`, `dmatInverse` |
| `dvecMin`, `dvecMax` | `dmatDiag`, `dmatPow`, `dmatFromRows`, `dmatFromCols` |
| `dvecDistance`, `dvecLerp` | `dmatGetRow`, `dmatGetCol`, `dmatIsSquare`, `dmatShape` |

## Performance

Native handles provide significant performance improvements over pure Nostos list operations:

| Operation | Native Handles | Pure Nostos | Speedup |
|-----------|---------------|-------------|---------|
| Vector add + sum (5 elem) | ~30μs | ~280μs | 9x |
| Dot product (10 elem) | ~2μs | ~22μs | 10x |
| Large vector ops (100 elem) | ~60μs | ~558μs | 9x |

The speedup comes from:
- No data copying between Nostos and Rust
- SIMD-accelerated operations in nalgebra
- Efficient memory layout for numeric data

## License

MIT
