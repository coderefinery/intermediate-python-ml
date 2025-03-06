# NumPy for Bioinformatics
## 3-Hour Workshop

## What is `numpy`?

* `NumPy` is short for "Numerical Python"
* Core python library for scientific computing
* Useful for processing large quantities of same-type data
* Foundation for:
  * Data manipulation, analysis and visualization libraries (`Pandas`, `Matplotlib`, `scipy`)
  * Machine learning libraries (`scikit-learn`, `TensorFlow`, `PyTorch`)
* NumPy operations are written in compiled C, significantly speeding up mathematical operations

## NumPy Arrays vs Python Lists

* Lists are data structures used to store collections of elements
* NumPy arrays enforce a single data type for all elements
* Benefits of NumPy arrays:
  * Homogeneity removes need for type checking during operations
  * Contiguous memory allocation (faster than Python's scattered storage)
  * Vectorization allows operations on entire arrays without loops
  * Rich set of mathematical functions and operations


## Creating NumPy Arrays (Hands-on)

### 1D Arrays from lists

```python
import numpy as np

# Create from list
py_list = list(range(1,5))
np_array = np.array(py_list)
print(np_array)  # Output: array([1, 2, 3, 4])
```

### 2D Arrays (matrices)

```python
# Create a 2D array
rows, cols = 3, 4
list_of_list = [[j for j in range(cols)] for i in range(rows)]
np_array = np.array(list_of_list)
print(np_array)
```

Output:
```
array([[0, 1, 2, 3],
       [0, 1, 2, 3],
       [0, 1, 2, 3]])
```

### Creating arrays from scratch

```python
# Range of values
np.arange(1, 10, 2)  # Output: array([1, 3, 5, 7, 9])

# Arrays of zeros
np.zeros((2, 2))     # Output: array([[0., 0.], [0., 0.]])

# Arrays of ones
np.ones(5)           # Output: array([1., 1., 1., 1., 1.])

# Random arrays
np.random.random((2, 2))  # Random values between 0 and 1
```

## Examining Array Structure (Exercise 1)

```python
import numpy as np

# Create an array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Examine attributes
print(f"Shape: {array_2d.shape}")   # (2, 3)
print(f"Dimensions: {array_2d.ndim}")  # 2
print(f"Size: {array_2d.size}")     # 6
print(f"Data type: {array_2d.dtype}")  # int64
```

## NumPy Data Types

### Python vs NumPy Data Types

| Feature           | Python Data Types                      | NumPy Data Types                   |
|-------------------|----------------------------------------|------------------------------------|
| Variety           | Broad (int, float, str, bool, etc.)    | Focused on numerical types         |
| Performance       | Slower for numerical operations        | Optimized for numerical calculations |
| Memory            | Variable overhead                      | Efficient, contiguous memory allocation |
| Homogeneity       | Can mix types                          | Single type across array           |

### Key NumPy Data Types for Bioinformatics

| NumPy Type   | Description                       | Use Case                        |
|--------------|-----------------------------------|----------------------------------|
| `np.int64`   | 64-bit signed integer             | Counts, indices                  |
| `np.float64` | Double-precision float            | Measurement values, probabilities |
| `np.bool_`   | Boolean (True/False)              | Filtering, masking               |
| `np.string_` | Fixed-length string               | Sequence data, labels            |
| `np.uint8`   | 8-bit unsigned integer (0-255)    | Compact storage for small values |

### Type Coercion (Exercise 2)

```python
# Type coercion example
a = np.array([1, 2, 3])         # int64
b = np.array([1.0, 2.0, 3.0])   # float64
result = a + b                  # float64

print(f"Type of a: {a.dtype}")
print(f"Type of b: {b.dtype}")
print(f"Type of result: {result.dtype}")
```

## Indexing and Slicing Arrays

### 1D Array Indexing

```python
data = np.array([1, 2, 3, 4, 5])

# Single element
print(data[0])     # 1

# Slicing
print(data[1:3])   # array([2, 3])
print(data[:3])    # array([1, 2, 3])
print(data[2:])    # array([3, 4, 5])
print(data[-2:])   # array([4, 5])
```

### 2D Array Indexing

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Single element
print(arr[1, 2])    # 6

# Row access
print(arr[1, :])    # array([4, 5, 6])

# Column access
print(arr[:, 2])    # array([3, 6, 9])

# Sub-array
print(arr[0:2, 1:3])
# array([[2, 3],
#        [5, 6]])
```

## Advanced Filtering with Boolean Masking (Exercise 3)

```python
data = np.array([1, 4, 2, 5, 3])

# Create a boolean mask
mask = data > 3
print(mask)  # array([False, True, False, True, False])

# Apply the mask
selected_data = data[mask]
print(selected_data)  # array([4, 5])

# Combining conditions
combined_mask = (data > 2) & (data < 5)
print(data[combined_mask])  # array([4, 3])
```

### Using np.where()

```python
data = np.arange(0, 20, 3)  # [0, 3, 6, 9, 12, 15, 18]

# Find indices of even values
indices = np.where(data % 2 == 0)
print(indices)  # (array([0, 2, 4, 6]),)

# Select values
print(data[indices])  # array([ 0,  6, 12, 18])

# Replace values conditionally
print(np.where(data % 2 == 0, data, 0))  
# array([ 0,  0,  6,  0, 12,  0, 18])
```

## Basic Array Operations

### Reshaping Arrays

```python
a = np.ones(6)  # array([1., 1., 1., 1., 1., 1.])

# Reshape to 2x3
b = a.reshape(2, 3)
print(b)
# array([[1., 1., 1.],
#        [1., 1., 1.]])

# Reshape to 3x2
c = a.reshape(3, 2)
print(c)
# array([[1., 1.],
#        [1., 1.],
#        [1., 1.]])
```

### Concatenation (Exercise 4)

```python
# 1D concatenation
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print(np.concatenate((a, b)))  # array([1, 2, 3, 4, 5, 6, 7, 8])

# 2D concatenation
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])

# Vertical concatenation (axis=0)
print(np.concatenate((x, y), axis=0))
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

# Horizontal concatenation (axis=1)
z = np.array([[7], [8]])
print(np.concatenate((x, z), axis=1))
# array([[1, 2, 7],
#        [3, 4, 8]])
```

### Summary Statistics

```python
data = np.array([[1, 2, 3], [4, 5, 6]])

# Total sum
print(np.sum(data))  # 21

# Column sums
print(np.sum(data, axis=0))  # array([5, 7, 9])

# Row sums
print(np.sum(data, axis=1))  # array([6, 15])

# Other statistics
print(np.min(data))  # 1
print(np.max(data))  # 6
print(np.mean(data))  # 3.5
```

## Vectorized Operations

### Basic Arithmetic

```python
import numpy as np
import time

# Compare loops vs. vectorization
numbers = np.arange(1, 1000001)

# With loop
start = time.time()
result_loop = []
for num in numbers:
    result_loop.append(num * 2)
loop_time = time.time() - start

# With vectorization
start = time.time()
result_vector = numbers * 2
vector_time = time.time() - start

print(f"Loop time: {loop_time:.6f} seconds")
print(f"Vector time: {vector_time:.6f} seconds")
print(f"Speedup: {loop_time/vector_time:.2f}x")
```

### Element-wise Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Addition
print(a + b)  # array([ 6,  8, 10, 12])

# Multiplication
print(a * b)  # array([ 5, 12, 21, 32])

# Exponentiation
print(a**2)   # array([ 1,  4,  9, 16])

# Functions
print(np.sqrt(a))  # array([1., 1.41421356, 1.73205081, 2.])
```

### Basic Broadcasting (Exercise 5)

```python
# Broadcasting allows operations between arrays of different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array
b = np.array([10, 20, 30])           # 1D array

# Broadcasting b to match a's shape
result = a + b
print(result)
# array([[11, 22, 33],
#        [14, 25, 36]])

# Scaling an array
print(a * 5)
# array([[ 5, 10, 15],
#        [20, 25, 30]])
```

## Bioinformatics Applications: Example Workflow

```python
# Example: DNA sequence analysis
import numpy as np

# DNA sequence as string
dna = "ATGCATGCATGCATGC"

# Convert to numpy array of characters
seq_array = np.array(list(dna))
print(seq_array)  # array(['A', 'T', 'G', 'C', ...], dtype='<U1')

# Count bases
unique, counts = np.unique(seq_array, return_counts=True)
base_counts = dict(zip(unique, counts))
print(base_counts)  # {'A': 4, 'C': 4, 'G': 4, 'T': 4}

# Find positions of G bases
g_positions = np.where(seq_array == 'G')[0]
print(g_positions)  # array([2, 6, 10, 14])

# Calculate GC content
gc_count = np.sum((seq_array == 'G') | (seq_array == 'C'))
gc_content = gc_count / len(seq_array) * 100
print(f"GC content: {gc_content:.1f}%")  # GC content: 50.0%
```

## Workshop Summary

### Key NumPy Features for Bioinformatics:
- Efficient storage and manipulation of large datasets
- Fast vectorized operations
- Boolean indexing for sequence analysis
- Aggregation and statistics calculation
- Integration with other scientific libraries

### Next Steps:
- Explore Pandas for structured biological data
- Learn Matplotlib/Seaborn for visualization
- Apply NumPy to your specific bioinformatics workflows
- Check more advanced NumPy features in the documentation

## Additional Resources:
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy for Bioinformatics Tutorials](https://www.tutorialspoint.com/biopython/index.htm)
- [Python for Biologists](https://pythonforbiologists.com/)