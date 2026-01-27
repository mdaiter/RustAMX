// AMX.swift - Swift bindings for Apple AMX coprocessor
// Provides safe, idiomatic Swift API with value semantics

import CAMX

// MARK: - AMX Version Detection

/// Detected AMX version based on Apple Silicon generation.
public enum AmxVersion: Int, Sendable {
    case unknown = 0
    case m1 = 1
    case m2 = 2
    case m3 = 3
    case m4 = 4
}

/// Detect AMX availability and version.
/// Result is cached after first call (thread-safe).
/// - Returns: The AMX version, or `nil` if not on Apple Silicon.
public func detect() -> AmxVersion? {
    let version = amx_detect()
    if version == AMX_VERSION_NONE {
        return nil
    }
    return AmxVersion(rawValue: Int(version.rawValue))
}

/// Check if AMX is available.
public var isAvailable: Bool {
    amx_is_available()
}

// MARK: - AMX Context (RAII Guard)

/// RAII guard for AMX state.
/// Enables AMX on creation, disables when deinitialized.
///
/// Usage:
/// ```swift
/// let context = try AmxContext()
/// // AMX is now enabled
/// // ... perform low-level AMX operations ...
/// // AMX automatically disabled when context goes out of scope
/// ```
public final class AmxContext: Sendable {
    
    /// Create an AMX context, enabling the AMX coprocessor.
    /// - Throws: `AmxError.notAvailable` if AMX is not available on this hardware.
    public init() throws {
        guard isAvailable else {
            throw AmxError.notAvailable
        }
        amx_set()
    }
    
    deinit {
        amx_clr()
    }
    
    /// Perform operations with AMX enabled.
    /// The context ensures AMX is enabled for the duration of the closure.
    ///
    /// - Parameter body: A closure that performs AMX operations.
    /// - Returns: The result of the closure.
    /// - Throws: Rethrows any error from the closure, or `AmxError.notAvailable`.
    public static func withContext<T>(_ body: () throws -> T) throws -> T {
        let context = try AmxContext()
        _ = context // Keep alive
        return try body()
    }
}

// MARK: - Errors

/// Errors that can occur during AMX operations.
public enum AmxError: Error, Sendable {
    /// AMX is not available on this hardware.
    case notAvailable
    /// Matrix dimensions don't match for the operation.
    case dimensionMismatch(expected: (Int, Int), got: (Int, Int))
    /// Memory allocation failed.
    case allocationFailed
}

// MARK: - Low-Level Operations

/// Namespace for low-level AMX operations.
/// All operations require an active `AmxContext`.
public enum Ops {
    
    /// Load 64 bytes into X register.
    /// - Parameters:
    ///   - data: Pointer to source data (must be at least 64 bytes, or 128 if pair=true)
    ///   - register: X register index (0-7)
    ///   - pair: If true, load 128 bytes into consecutive registers
    @inlinable
    public static func loadX(_ data: UnsafeRawPointer, register: UInt64, pair: Bool = false) {
        amx_load_x(data, register, pair)
    }
    
    /// Load 64 bytes into Y register.
    @inlinable
    public static func loadY(_ data: UnsafeRawPointer, register: UInt64, pair: Bool = false) {
        amx_load_y(data, register, pair)
    }
    
    /// Load 64 bytes into Z register row.
    @inlinable
    public static func loadZ(_ data: UnsafeRawPointer, row: UInt64, pair: Bool = false) {
        amx_load_z(data, row, pair)
    }
    
    /// Store 64 bytes from X register.
    @inlinable
    public static func storeX(_ data: UnsafeMutableRawPointer, register: UInt64, pair: Bool = false) {
        amx_store_x(data, register, pair)
    }
    
    /// Store 64 bytes from Y register.
    @inlinable
    public static func storeY(_ data: UnsafeMutableRawPointer, register: UInt64, pair: Bool = false) {
        amx_store_y(data, register, pair)
    }
    
    /// Store 64 bytes from Z register row.
    @inlinable
    public static func storeZ(_ data: UnsafeMutableRawPointer, row: UInt64, pair: Bool = false) {
        amx_store_z(data, row, pair)
    }
    
    /// Fused multiply-add for f32: Z += X * Y
    /// - Parameters:
    ///   - xOffset: Byte offset into X register file (0-511)
    ///   - yOffset: Byte offset into Y register file (0-511)
    ///   - zRow: Z register row (0-63)
    ///   - vectorMode: false for outer product, true for pointwise
    @inlinable
    public static func fma32(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_fma32_op(xOffset, yOffset, zRow, vectorMode)
    }
    
    /// Fused multiply-add for f64
    @inlinable
    public static func fma64(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_fma64_op(xOffset, yOffset, zRow, vectorMode)
    }
    
    /// Fused multiply-add for f16
    @inlinable
    public static func fma16(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_fma16_op(xOffset, yOffset, zRow, vectorMode)
    }
    
    /// Fused multiply-subtract for f32: Z -= X * Y
    @inlinable
    public static func fms32(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_fms32_op(xOffset, yOffset, zRow, vectorMode)
    }
    
    /// Fused multiply-subtract for f64
    @inlinable
    public static func fms64(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_fms64_op(xOffset, yOffset, zRow, vectorMode)
    }
    
    /// Fused multiply-subtract for f16
    @inlinable
    public static func fms16(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_fms16_op(xOffset, yOffset, zRow, vectorMode)
    }
    
    /// Integer multiply-accumulate for i16
    @inlinable
    public static func mac16(xOffset: UInt64, yOffset: UInt64, zRow: UInt64, vectorMode: Bool = false) {
        amx_mac16_op(xOffset, yOffset, zRow, vectorMode)
    }
}
