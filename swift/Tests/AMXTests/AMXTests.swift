import XCTest
@testable import AMX

final class AMXTests: XCTestCase {
    
    // MARK: - Detection Tests
    
    func testDetection() {
        let version = detect()
        print("AMX version: \(String(describing: version))")
        
        // On Apple Silicon, should return a version
        #if arch(arm64)
        XCTAssertNotNil(version, "AMX should be available on Apple Silicon")
        #endif
    }
    
    func testIsAvailable() {
        #if arch(arm64)
        XCTAssertTrue(isAvailable, "AMX should be available on Apple Silicon")
        #endif
    }
    
    // MARK: - Matrix Creation Tests
    
    func testMatrixZeros() {
        let m = Matrix.zeros(3, 4)
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.cols, 4)
        XCTAssertTrue(m.shape == (3, 4))
        
        m.withUnsafeBufferPointer { ptr in
            XCTAssertTrue(ptr.allSatisfy { $0 == 0 })
        }
    }
    
    func testMatrixFill() {
        let m = Matrix.fill(2, 2, value: 5.0)
        // Check logical elements (not padding) are all 5.0
        for i in 0..<m.rows {
            for j in 0..<m.cols {
                XCTAssertEqual(m[i, j], 5.0)
            }
        }
    }
    
    func testMatrixIdentity() {
        let m = Matrix.identity(3)
        XCTAssertEqual(m[0, 0], 1.0)
        XCTAssertEqual(m[0, 1], 0.0)
        XCTAssertEqual(m[1, 1], 1.0)
        XCTAssertEqual(m[2, 2], 1.0)
        XCTAssertEqual(m[1, 2], 0.0)
    }
    
    func testMatrixFromArray() {
        let data: [Float] = [1, 2, 3, 4, 5, 6]
        let m = Matrix(rows: 2, cols: 3, data: data)
        
        XCTAssertEqual(m[0, 0], 1)
        XCTAssertEqual(m[0, 2], 3)
        XCTAssertEqual(m[1, 0], 4)
        XCTAssertEqual(m[1, 2], 6)
    }
    
    // MARK: - Element Access Tests
    
    func testSubscript() {
        var m = Matrix.zeros(2, 2)
        m[0, 0] = 1.0
        m[0, 1] = 2.0
        m[1, 0] = 3.0
        m[1, 1] = 4.0
        
        XCTAssertEqual(m[0, 0], 1.0)
        XCTAssertEqual(m[0, 1], 2.0)
        XCTAssertEqual(m[1, 0], 3.0)
        XCTAssertEqual(m[1, 1], 4.0)
    }
    
    // MARK: - Copy-on-Write Tests
    
    func testCopyOnWrite() {
        let a = Matrix.fill(2, 2, value: 1.0)
        var b = a  // Shallow copy
        
        // Before mutation, should share storage
        XCTAssertEqual(a[0, 0], 1.0)
        XCTAssertEqual(b[0, 0], 1.0)
        
        // Mutation should trigger copy
        b[0, 0] = 99.0
        
        // a should be unchanged
        XCTAssertEqual(a[0, 0], 1.0)
        XCTAssertEqual(b[0, 0], 99.0)
    }
    
    // MARK: - Matrix Operations Tests
    
    func testTranspose() {
        let m = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
        let t = m.transposed()
        
        XCTAssertTrue(t.shape == (3, 2))
        XCTAssertEqual(t[0, 0], 1)
        XCTAssertEqual(t[0, 1], 4)
        XCTAssertEqual(t[1, 0], 2)
        XCTAssertEqual(t[2, 1], 6)
    }
    
    func testAdd() {
        let a = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
        let c = a + b
        
        XCTAssertEqual(c.toArray(), [6, 8, 10, 12])
    }
    
    func testSubtract() {
        let a = Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
        let b = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let c = a - b
        
        XCTAssertEqual(c.toArray(), [4, 4, 4, 4])
    }
    
    func testScale() {
        let a = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = a * 2.0
        
        XCTAssertEqual(b.toArray(), [2, 4, 6, 8])
    }
    
    func testNegate() {
        let a = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = -a
        
        XCTAssertEqual(b.toArray(), [-1, -2, -3, -4])
    }
    
    // MARK: - Matrix Multiplication Tests
    
    func testMatmulSmall() {
        // 2x2 @ 2x2 (will use naive fallback)
        let a = Matrix(rows: 2, cols: 2, data: [1, 2, 3, 4])
        let b = Matrix(rows: 2, cols: 2, data: [5, 6, 7, 8])
        let c = a * b
        
        // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        XCTAssertEqual(c.toArray(), [19, 22, 43, 50])
    }
    
    func testMatmulIdentity() {
        guard isAvailable else {
            print("Skipping AMX test - not available")
            return
        }
        
        let n = 64
        let a = Matrix.identity(n)
        let b = Matrix(rows: n, cols: n, data: (0..<n*n).map { Float($0 % n) })
        let c = a * b
        
        // Identity * B = B
        c.withUnsafeBufferPointer { cPtr in
            b.withUnsafeBufferPointer { bPtr in
                for i in 0..<n*n {
                    XCTAssertEqual(cPtr[i], bPtr[i], accuracy: 1e-5,
                                   "Mismatch at \(i): \(cPtr[i]) != \(bPtr[i])")
                }
            }
        }
    }
    
    func testMatmulLarge() {
        guard isAvailable else {
            print("Skipping AMX test - not available")
            return
        }
        
        let n = 128
        let a = Matrix.fill(n, n, value: 1.0)
        let b = Matrix.fill(n, n, value: 2.0)
        let c = a * b
        
        // Each element should be n * 1.0 * 2.0 = 2n
        let expected = Float(n) * 2.0
        c.withUnsafeBufferPointer { ptr in
            for (i, val) in ptr.enumerated() {
                XCTAssertEqual(val, expected, accuracy: 1e-3,
                               "Mismatch at \(i): \(val) != \(expected)")
            }
        }
    }
    
    // MARK: - AMX Context Tests
    
    func testAmxContext() throws {
        guard isAvailable else {
            print("Skipping AMX context test - not available")
            return
        }
        
        // Test RAII pattern
        do {
            let context = try AmxContext()
            _ = context  // Keep alive
            // AMX should be enabled here
        }
        // AMX should be disabled after scope
    }
    
    func testAmxContextWithClosure() throws {
        guard isAvailable else {
            print("Skipping AMX context test - not available")
            return
        }
        
        let result = try AmxContext.withContext {
            // AMX is enabled here
            return 42
        }
        XCTAssertEqual(result, 42)
    }
    
    // MARK: - Performance Tests
    
    func testMatmulPerformance() throws {
        guard isAvailable else {
            print("Skipping performance test - AMX not available")
            return
        }
        
        let n = 256
        let a = Matrix.fill(n, n, value: 1.0)
        let b = Matrix.fill(n, n, value: 2.0)
        
        measure {
            let _ = a * b
        }
    }
}
