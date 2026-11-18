const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c_img.h"),
        .target = target,
        .optimize = optimize,
    });
    const c_img_mod = translate_c.createModule();

    const translate_wgpu = b.addTranslateC(.{
        .root_source_file = b.path("src/c_wgpu.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_wgpu.addIncludePath(b.path("third_party/wgpu-native/include"));
    const c_wgpu_mod = translate_wgpu.createModule();

    const translate_ort = b.addTranslateC(.{
        .root_source_file = b.path("src/c_ort.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_ort.addIncludePath(b.path("third_party/onnxruntime/include"));
    const c_ort_mod = translate_ort.createModule();

    const mod = b.addModule("3d_reconstruction", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "c_img", .module = c_img_mod },
            .{ .name = "c_wgpu", .module = c_wgpu_mod },
            .{ .name = "c_ort", .module = c_ort_mod },
        },
    });
    mod.linkSystemLibrary("png", .{});
    mod.linkSystemLibrary("jpeg", .{});
    mod.addLibraryPath(b.path("third_party/wgpu-native/lib"));
    mod.addRPath(b.path("third_party/wgpu-native/lib"));
    mod.linkSystemLibrary("wgpu_native", .{});
    mod.addLibraryPath(b.path("third_party/onnxruntime/lib"));
    mod.addRPath(b.path("third_party/onnxruntime/lib"));
    mod.linkSystemLibrary("onnxruntime", .{});

    const exe = b.addExecutable(.{
        .name = "3d_reconstruction",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "3d_reconstruction", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "3d_reconstruction", .module = mod },
            },
        }),
    });
    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run performance benchmarks");
    bench_step.dependOn(&run_bench.step);

    const translate_gl = b.addTranslateC(.{
        .root_source_file = b.path("src/c_gl.h"),
        .target = target,
        .optimize = optimize,
    });
    const c_gl_mod = translate_gl.createModule();

    const viewer_exe = b.addExecutable(.{
        .name = "sfm_viewer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/viewer.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "3d_reconstruction", .module = mod },
                .{ .name = "c_gl", .module = c_gl_mod },
            },
        }),
    });
    viewer_exe.root_module.linkSystemLibrary("glfw", .{});
    viewer_exe.root_module.linkSystemLibrary("GL", .{});
    viewer_exe.root_module.linkSystemLibrary("X11", .{});
    b.installArtifact(viewer_exe);

    const run_viewer = b.addRunArtifact(viewer_exe);
    run_viewer.step.dependOn(b.getInstallStep());
    run_viewer.addPassthruArgs();
    const viewer_step = b.step("viewer", "Run native OpenGL mesh & camera viewer");
    viewer_step.dependOn(&run_viewer.step);
}
