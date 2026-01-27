// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AMX",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "AMX",
            targets: ["AMX"]),
    ],
    targets: [
        // C library with AMX inline assembly
        .target(
            name: "CAMX",
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(["-O3", "-fno-strict-aliasing"])
            ]
        ),
        // Swift wrapper with idiomatic API
        .target(
            name: "AMX",
            dependencies: ["CAMX"]
        ),
        // Tests
        .testTarget(
            name: "AMXTests",
            dependencies: ["AMX"]
        ),
    ]
)
