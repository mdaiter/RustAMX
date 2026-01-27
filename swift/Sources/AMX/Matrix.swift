// Matrix.swift - High-level Matrix type with value semantics and COW
// Optimized for AMX: 64-byte aligned storage with padded stride

import CAMX
import Foundation

// MARK: - Constants

private let AMX_ALIGN = 64
private let AMX_TILE_DIM = 16

@inline(__always)
private func roundUpTile(_ n: Int) -> Int {
    (n + AMX_TILE_DIM - 1) & ~(AMX_TILE_DIM - 1)
}

// MARK: - Matrix Storage (Reference Type for COW)

@usableFromInline
internal final class MatrixStorage {
    @usableFromInline let ptr: UnsafeMutableRawPointer
    @usableFromInline let rows: Int
    @usableFromInline let cols: Int
    @usableFromInline let stride: Int
    
    @inline(__always) @usableFromInline
    var data: UnsafeMutablePointer<Float> {
        ptr.assumingMemoryBound(to: Float.self)
    }
    
    @usableFromInline
    init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.stride = roundUpTile(cols)
        
        let byteCount = rows * self.stride * MemoryLayout<Float>.size
        self.ptr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: AMX_ALIGN)
        memset(self.ptr, 0, byteCount)
    }
    
    @usableFromInline
    init(rows: Int, cols: Int, stride: Int, ptr: UnsafeMutableRawPointer) {
        self.rows = rows
        self.cols = cols
        self.stride = stride
        self.ptr = ptr
    }
    
    deinit {
        ptr.deallocate()
    }
    
    @usableFromInline
    func copy() -> MatrixStorage {
        let byteCount = rows * stride * MemoryLayout<Float>.size
        let newPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: AMX_ALIGN)
        memcpy(newPtr, ptr, byteCount)
        return MatrixStorage(rows: rows, cols: cols, stride: stride, ptr: newPtr)
    }
    
    @inline(__always) @usableFromInline
    func get(_ row: Int, _ col: Int) -> Float {
        (data + row * stride + col).pointee
    }
    
    @inline(__always) @usableFromInline
    func set(_ row: Int, _ col: Int, _ value: Float) {
        (data + row * stride + col).pointee = value
    }
    
    @inline(__always) @usableFromInline
    func rowPtr(_ row: Int) -> UnsafeMutablePointer<Float> {
        data + row * stride
    }
}

// MARK: - Matrix (Value Type)

public struct Matrix: Sendable {
    
    @usableFromInline
    nonisolated(unsafe) internal var storage: MatrixStorage
    
    // MARK: - Initializers
    
    public init(rows: Int, cols: Int) {
        precondition(rows > 0 && cols > 0, "Matrix dimensions must be positive")
        self.storage = MatrixStorage(rows: rows, cols: cols)
    }
    
    public static func zeros(_ rows: Int, _ cols: Int) -> Matrix {
        Matrix(rows: rows, cols: cols)
    }
    
    public static func fill(_ rows: Int, _ cols: Int, value: Float) -> Matrix {
        let m = Matrix(rows: rows, cols: cols)
        let p = m.storage.data
        let stride = m.storage.stride
        for i in 0..<rows {
            let row = p + i * stride
            for j in 0..<cols {
                (row + j).pointee = value
            }
        }
        return m
    }
    
    public static func identity(_ n: Int) -> Matrix {
        let m = Matrix(rows: n, cols: n)
        let p = m.storage.data
        let stride = m.storage.stride
        for i in 0..<n {
            (p + i * stride + i).pointee = 1.0
        }
        return m
    }
    
    public init(rows: Int, cols: Int, data: [Float]) {
        precondition(data.count == rows * cols, "Data length must equal rows * cols")
        self.storage = MatrixStorage(rows: rows, cols: cols)
        let dst = storage.data
        let dstStride = storage.stride
        data.withUnsafeBufferPointer { src in
            for i in 0..<rows {
                memcpy(dst + i * dstStride, src.baseAddress! + i * cols, cols * MemoryLayout<Float>.size)
            }
        }
    }
    
    public init<C: Collection>(rows: Int, cols: Int, data: C) where C.Element == Float {
        precondition(data.count == rows * cols, "Data length must equal rows * cols")
        self.storage = MatrixStorage(rows: rows, cols: cols)
        let dst = storage.data
        let dstStride = storage.stride
        var idx = data.startIndex
        for i in 0..<rows {
            let row = dst + i * dstStride
            for j in 0..<cols {
                (row + j).pointee = data[idx]
                idx = data.index(after: idx)
            }
        }
    }
    
    // MARK: - Properties
    
    @inlinable public var rows: Int { storage.rows }
    @inlinable public var cols: Int { storage.cols }
    @inlinable public var stride: Int { storage.stride }
    @inlinable public var shape: (Int, Int) { (storage.rows, storage.cols) }
    @inlinable public var count: Int { storage.rows * storage.cols }
    @inlinable public var isSquare: Bool { storage.rows == storage.cols }
    
    // MARK: - Copy-on-Write
    
    @usableFromInline
    internal mutating func ensureUnique() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage.copy()
        }
    }
    
    // MARK: - Data Access
    
    public func toArray() -> [Float] {
        var result = [Float](repeating: 0, count: rows * cols)
        let src = storage.data
        let srcStride = storage.stride
        result.withUnsafeMutableBufferPointer { dst in
            for i in 0..<rows {
                memcpy(dst.baseAddress! + i * cols, src + i * srcStride, cols * MemoryLayout<Float>.size)
            }
        }
        return result
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        let buf = UnsafeBufferPointer(start: storage.data, count: storage.rows * storage.stride)
        return try body(buf)
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        ensureUnique()
        var buf = UnsafeMutableBufferPointer(start: storage.data, count: storage.rows * storage.stride)
        return try body(&buf)
    }
    
    @inlinable
    public func withRowPointer<R>(row: Int, _ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        try body(storage.rowPtr(row))
    }
    
    // MARK: - Element Access
    
    @inlinable
    public subscript(row: Int, col: Int) -> Float {
        get {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            return storage.get(row, col)
        }
        set {
            precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
            ensureUnique()
            storage.set(row, col, newValue)
        }
    }
    
    // MARK: - Matrix Operations
    
    public func matmul(_ other: Matrix) -> Matrix {
        precondition(cols == other.rows,
            "Matrix dimensions don't match: (\(rows), \(cols)) * (\(other.rows), \(other.cols))")
        
        var result = Matrix(rows: rows, cols: other.cols)
        
        if isAvailable && rows >= AMX_TILE_DIM && other.cols >= AMX_TILE_DIM {
            matmulAMX(self, other, &result)
        } else {
            matmulNaive(self, other, &result)
        }
        
        return result
    }
    
    public func transposed() -> Matrix {
        let result = Matrix(rows: cols, cols: rows)
        let src = storage.data
        let dst = result.storage.data
        let srcStride = storage.stride
        let dstStride = result.storage.stride
        
        for i in 0..<rows {
            let srcRow = src + i * srcStride
            for j in 0..<cols {
                (dst + j * dstStride + i).pointee = (srcRow + j).pointee
            }
        }
        return result
    }
    
    public func adding(_ other: Matrix) -> Matrix {
        precondition(shape == other.shape, "Matrix shapes must match")
        let result = Matrix(rows: rows, cols: cols)
        let ap = storage.data
        let bp = other.storage.data
        let cp = result.storage.data
        let aStride = storage.stride
        let bStride = other.storage.stride
        let cStride = result.storage.stride
        
        for i in 0..<rows {
            let aRow = ap + i * aStride
            let bRow = bp + i * bStride
            let cRow = cp + i * cStride
            for j in 0..<cols {
                (cRow + j).pointee = (aRow + j).pointee + (bRow + j).pointee
            }
        }
        return result
    }
    
    public func subtracting(_ other: Matrix) -> Matrix {
        precondition(shape == other.shape, "Matrix shapes must match")
        let result = Matrix(rows: rows, cols: cols)
        let ap = storage.data
        let bp = other.storage.data
        let cp = result.storage.data
        let aStride = storage.stride
        let bStride = other.storage.stride
        let cStride = result.storage.stride
        
        for i in 0..<rows {
            let aRow = ap + i * aStride
            let bRow = bp + i * bStride
            let cRow = cp + i * cStride
            for j in 0..<cols {
                (cRow + j).pointee = (aRow + j).pointee - (bRow + j).pointee
            }
        }
        return result
    }
    
    public func scaled(by scalar: Float) -> Matrix {
        let result = Matrix(rows: rows, cols: cols)
        let src = storage.data
        let dst = result.storage.data
        let srcStride = storage.stride
        let dstStride = result.storage.stride
        
        for i in 0..<rows {
            let srcRow = src + i * srcStride
            let dstRow = dst + i * dstStride
            for j in 0..<cols {
                (dstRow + j).pointee = (srcRow + j).pointee * scalar
            }
        }
        return result
    }
    
    public func negated() -> Matrix {
        scaled(by: -1)
    }
}

// MARK: - Matrix Multiplication Implementations

private func matmulNaive(_ a: Matrix, _ b: Matrix, _ c: inout Matrix) {
    let m = a.rows
    let k = a.cols
    let n = b.cols
    
    let ap = a.storage.data
    let bp = b.storage.data
    let cp = c.storage.data
    let aStride = a.stride
    let bStride = b.stride
    let cStride = c.stride
    
    memset(c.storage.ptr, 0, c.rows * cStride * MemoryLayout<Float>.size)
    
    for i in 0..<m {
        let aRow = ap + i * aStride
        let cRow = cp + i * cStride
        for kk in 0..<k {
            let aik = (aRow + kk).pointee
            let bRow = bp + kk * bStride
            for j in 0..<n {
                let cPtr = cRow + j
                cPtr.pointee += aik * (bRow + j).pointee
            }
        }
    }
}

private func matmulAMX(_ a: Matrix, _ b: Matrix, _ c: inout Matrix) {
    let m = a.rows
    let k = a.cols
    let n = b.cols
    
    let ap = a.storage.data
    let bp = b.storage.data
    let cp = c.storage.data
    let aStride = a.stride
    let bStride = b.stride
    let cStride = c.stride
    
    memset(c.storage.ptr, 0, c.rows * cStride * MemoryLayout<Float>.size)
    
    // Thread-local aligned buffer for A column gathering
    var aCol: (Float, Float, Float, Float, Float, Float, Float, Float,
               Float, Float, Float, Float, Float, Float, Float, Float) =
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    var zeros: (Float, Float, Float, Float, Float, Float, Float, Float,
                Float, Float, Float, Float, Float, Float, Float, Float) =
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    
    amx_set()
    defer { amx_clr() }
    
    for i in Swift.stride(from: 0, to: m, by: AMX_TILE_DIM) {
        let tileRows = min(AMX_TILE_DIM, m - i)
        let aTile = ap + i * aStride
        let cTileBase = cp + i * cStride
        
        for j in Swift.stride(from: 0, to: n, by: AMX_TILE_DIM) {
            let cTile = cTileBase + j
            
            // Load C tile into Z registers
            for ti in 0..<tileRows {
                amx_load_z(cTile + ti * cStride, UInt64(ti * 4), false)
            }
            withUnsafePointer(to: &zeros) { ptr in
                for ti in tileRows..<AMX_TILE_DIM {
                    amx_load_z(ptr, UInt64(ti * 4), false)
                }
            }
            
            // Hot loop: accumulate over k
            let bCol = bp + j
            
            for kk in 0..<k {
                // Gather A column
                withUnsafeMutablePointer(to: &aCol) { ptr in
                    let aColPtr = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: Float.self)
                    if tileRows == AMX_TILE_DIM {
                        var aPtr = aTile + kk
                        for ti in 0..<16 {
                            (aColPtr + ti).pointee = aPtr.pointee
                            aPtr += aStride
                        }
                    } else {
                        memset(ptr, 0, 64)
                        var aPtr = aTile + kk
                        for ti in 0..<tileRows {
                            (aColPtr + ti).pointee = aPtr.pointee
                            aPtr += aStride
                        }
                    }
                    amx_load_y(ptr, 0, false)
                }
                
                // Direct load B row
                amx_load_x(bCol + kk * bStride, 0, false)
                
                // FMA
                amx_fma32_op(0, 0, 0, false)
            }
            
            // Store C tile
            for ti in 0..<tileRows {
                amx_store_z(UnsafeMutableRawPointer(cTile + ti * cStride), UInt64(ti * 4), false)
            }
        }
    }
}

// MARK: - Operators

extension Matrix {
    public static func * (lhs: Matrix, rhs: Matrix) -> Matrix { lhs.matmul(rhs) }
    public static func + (lhs: Matrix, rhs: Matrix) -> Matrix { lhs.adding(rhs) }
    public static func - (lhs: Matrix, rhs: Matrix) -> Matrix { lhs.subtracting(rhs) }
    public static func * (lhs: Matrix, rhs: Float) -> Matrix { lhs.scaled(by: rhs) }
    public static func * (lhs: Float, rhs: Matrix) -> Matrix { rhs.scaled(by: lhs) }
    public static prefix func - (matrix: Matrix) -> Matrix { matrix.negated() }
}

// MARK: - Equatable

extension Matrix: Equatable {
    public static func == (lhs: Matrix, rhs: Matrix) -> Bool {
        guard lhs.shape == rhs.shape else { return false }
        let lp = lhs.storage.data
        let rp = rhs.storage.data
        let lStride = lhs.stride
        let rStride = rhs.stride
        for i in 0..<lhs.rows {
            let lRow = lp + i * lStride
            let rRow = rp + i * rStride
            for j in 0..<lhs.cols {
                if (lRow + j).pointee != (rRow + j).pointee { return false }
            }
        }
        return true
    }
}

// MARK: - CustomStringConvertible

extension Matrix: CustomStringConvertible {
    public var description: String {
        var preview: [Float] = []
        for i in 0..<min(2, rows) {
            for j in 0..<min(2, cols) {
                preview.append(storage.get(i, j))
            }
        }
        return "Matrix(\(rows)x\(cols), \(preview)...)"
    }
}

// MARK: - CustomDebugStringConvertible

extension Matrix: CustomDebugStringConvertible {
    public var debugDescription: String {
        var result = "Matrix(\(rows)x\(cols), stride=\(stride)):\n"
        for i in 0..<min(rows, 10) {
            result += "  ["
            for j in 0..<min(cols, 10) {
                if j > 0 { result += ", " }
                result += NSString(format: "%.4f", storage.get(i, j)) as String
            }
            if cols > 10 { result += ", ..." }
            result += "]\n"
        }
        if rows > 10 { result += "  ...\n" }
        return result
    }
}
