const std = @import("std");
const Io = std.Io;
const sfm = @import("3d_reconstruction");
const vm = sfm.viewer_math;
const Vec3 = sfm.vec.Vec3;
const CameraPose = sfm.lie.CameraPose;
const c = @import("c_gl");

var g_scroll_y: f64 = 0.0;

fn scrollCallback(_: ?*c.GLFWwindow, _: f64, yoffset: f64) callconv(.c) void {
    g_scroll_y += yoffset;
}

const Vertex3D = extern struct {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    g: f32,
    b: f32,
};

const Vertex2DColor = extern struct {
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

const Vertex2DTex = extern struct {
    x: f32,
    y: f32,
    u: f32,
    v: f32,
};

const vs_3d_src: [:0]const u8 =
    \\#version 330 core
    \\layout (location = 0) in vec3 aPos;
    \\layout (location = 1) in vec3 aColor;
    \\uniform mat4 uMVP;
    \\out vec3 vColor;
    \\void main() {
    \\    gl_Position = uMVP * vec4(aPos, 1.0);
    \\    vColor = aColor;
    \\}
;

const fs_3d_src: [:0]const u8 =
    \\#version 330 core
    \\in vec3 vColor;
    \\out vec4 FragColor;
    \\void main() {
    \\    FragColor = vec4(vColor, 1.0);
    \\}
;

const vs_2d_col_src: [:0]const u8 =
    \\#version 330 core
    \\layout (location = 0) in vec2 aPos;
    \\layout (location = 1) in vec4 aColor;
    \\out vec4 vColor;
    \\void main() {
    \\    gl_Position = vec4(aPos, 0.0, 1.0);
    \\    vColor = aColor;
    \\}
;

const fs_2d_col_src: [:0]const u8 =
    \\#version 330 core
    \\in vec4 vColor;
    \\out vec4 FragColor;
    \\void main() {
    \\    FragColor = vColor;
    \\}
;

const vs_2d_tex_src: [:0]const u8 =
    \\#version 330 core
    \\layout (location = 0) in vec2 aPos;
    \\layout (location = 1) in vec2 aUV;
    \\out vec2 vUV;
    \\void main() {
    \\    gl_Position = vec4(aPos, 0.0, 1.0);
    \\    vUV = aUV;
    \\}
;

const fs_2d_tex_src: [:0]const u8 =
    \\#version 330 core
    \\in vec2 vUV;
    \\uniform sampler2D uTex;
    \\uniform float uAlpha;
    \\out vec4 FragColor;
    \\void main() {
    \\    vec3 col = texture(uTex, vUV).rgb;
    \\    FragColor = vec4(col, uAlpha);
    \\}
;

fn compileShader(shader_type: c.GLenum, src: [:0]const u8) !c.GLuint {
    const sh = c.glCreateShader(shader_type);
    var ptr: [*c]const u8 = src.ptr;
    var len: c.GLint = @intCast(src.len);
    c.glShaderSource(sh, 1, &ptr, &len);
    c.glCompileShader(sh);
    var ok: c.GLint = 0;
    c.glGetShaderiv(sh, c.GL_COMPILE_STATUS, &ok);
    if (ok == 0) {
        var buf: [512]u8 = undefined;
        var out_len: c.GLsizei = 0;
        c.glGetShaderInfoLog(sh, 512, &out_len, &buf);
        std.log.err("GL shader compile error: {s}", .{buf[0..@intCast(out_len)]});
        return error.ShaderCompileFailed;
    }
    return sh;
}

fn linkProgram(vs_src: [:0]const u8, fs_src: [:0]const u8) !c.GLuint {
    const vs = try compileShader(c.GL_VERTEX_SHADER, vs_src);
    defer c.glDeleteShader(vs);
    const fs = try compileShader(c.GL_FRAGMENT_SHADER, fs_src);
    defer c.glDeleteShader(fs);

    const prog = c.glCreateProgram();
    c.glAttachShader(prog, vs);
    c.glAttachShader(prog, fs);
    c.glLinkProgram(prog);

    var ok: c.GLint = 0;
    c.glGetProgramiv(prog, c.GL_LINK_STATUS, &ok);
    if (ok == 0) {
        var buf: [512]u8 = undefined;
        var out_len: c.GLsizei = 0;
        c.glGetProgramInfoLog(prog, 512, &out_len, &buf);
        std.log.err("GL program link error: {s}", .{buf[0..@intCast(out_len)]});
        return error.ProgramLinkFailed;
    }
    return prog;
}

const Ui2DBatch = struct {
    verts: std.ArrayList(Vertex2DColor) = .empty,

    fn deinit(self: *Ui2DBatch, allocator: std.mem.Allocator) void {
        self.verts.deinit(allocator);
    }

    fn clear(self: *Ui2DBatch) void {
        self.verts.clearRetainingCapacity();
    }

    fn addRect(
        self: *Ui2DBatch,
        allocator: std.mem.Allocator,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        win_w: f32,
        win_h: f32,
        rgba: [4]f32,
    ) !void {
        const x0 = 2.0 * x / win_w - 1.0;
        const x1 = 2.0 * (x + w) / win_w - 1.0;
        const y0 = 1.0 - 2.0 * y / win_h;
        const y1 = 1.0 - 2.0 * (y + h) / win_h;

        const r = rgba[0];
        const g = rgba[1];
        const b = rgba[2];
        const a = rgba[3];

        try self.verts.appendSlice(allocator, &.{
            .{ .x = x0, .y = y0, .r = r, .g = g, .b = b, .a = a },
            .{ .x = x1, .y = y0, .r = r, .g = g, .b = b, .a = a },
            .{ .x = x1, .y = y1, .r = r, .g = g, .b = b, .a = a },
            .{ .x = x0, .y = y0, .r = r, .g = g, .b = b, .a = a },
            .{ .x = x1, .y = y1, .r = r, .g = g, .b = b, .a = a },
            .{ .x = x0, .y = y1, .r = r, .g = g, .b = b, .a = a },
        });
    }

    fn addRectBorder(
        self: *Ui2DBatch,
        allocator: std.mem.Allocator,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        t: f32,
        win_w: f32,
        win_h: f32,
        rgba: [4]f32,
    ) !void {
        try self.addRect(allocator, x, y, w, t, win_w, win_h, rgba);
        try self.addRect(allocator, x, y + h - t, w, t, win_w, win_h, rgba);
        try self.addRect(allocator, x, y, t, h, win_w, win_h, rgba);
        try self.addRect(allocator, x + w - t, y, t, h, win_w, win_h, rgba);
    }

    fn addText(
        self: *Ui2DBatch,
        allocator: std.mem.Allocator,
        x: f32,
        y: f32,
        scale: f32,
        text: []const u8,
        win_w: f32,
        win_h: f32,
        rgba: [4]f32,
    ) !void {
        var cx = x;
        for (text) |ch| {
            if (ch != ' ') {
                const rows = vm.glyph5x7(ch);
                for (rows, 0..) |row_bits, ry| {
                    var rx: usize = 0;
                    while (rx < 5) : (rx += 1) {
                        const mask: u8 = @as(u8, 0x10) >> @intCast(rx);
                        if ((row_bits & mask) != 0) {
                            try self.addRect(
                                allocator,
                                cx + @as(f32, @floatFromInt(rx)) * scale,
                                y + @as(f32, @floatFromInt(ry)) * scale,
                                scale,
                                scale,
                                win_w,
                                win_h,
                                rgba,
                            );
                        }
                    }
                }
            }
            cx += 6.0 * scale;
        }
    }
};

fn appendCameraGizmoLines(
    allocator: std.mem.Allocator,
    lines: *std.ArrayList(Vertex3D),
    tris: *std.ArrayList(Vertex3D),
    pose: CameraPose,
    k: sfm.mat.Mat3,
    img_w: f64,
    img_h: f64,
    depth: f64,
    line_col: [3]f32,
    body_col: [3]f32,
) !void {
    const c_pos = pose.cameraCenter();
    const rt = pose.r.transpose();
    const fx = k.m[0][0];
    const fy = k.m[1][1];
    const cx = k.m[0][2];
    const cy = k.m[1][2];

    const left_x = (0.0 - cx) / fx * depth;
    const right_x = (img_w - cx) / fx * depth;
    const top_y = (0.0 - cy) / fy * depth;
    const bot_y = (img_h - cy) / fy * depth;

    const corners_cam = [4]Vec3{
        Vec3.init(left_x, top_y, depth),
        Vec3.init(right_x, top_y, depth),
        Vec3.init(right_x, bot_y, depth),
        Vec3.init(left_x, bot_y, depth),
    };
    var corners_w: [4]Vec3 = undefined;
    for (corners_cam, 0..) |cc, idx| {
        corners_w[idx] = c_pos.add(rt.mulVec(cc));
    }

    const body_r = depth * 0.28;
    const body_pts = [6]Vec3{
        c_pos.add(Vec3.init(body_r, 0, 0)),
        c_pos.add(Vec3.init(-body_r, 0, 0)),
        c_pos.add(Vec3.init(0, body_r, 0)),
        c_pos.add(Vec3.init(0, -body_r, 0)),
        c_pos.add(Vec3.init(0, 0, body_r)),
        c_pos.add(Vec3.init(0, 0, -body_r)),
    };
    const oct_faces = [8][3]usize{
        .{ 0, 2, 4 }, .{ 2, 1, 4 }, .{ 1, 3, 4 }, .{ 3, 0, 4 },
        .{ 2, 0, 5 }, .{ 1, 2, 5 }, .{ 3, 1, 5 }, .{ 0, 3, 5 },
    };
    for (oct_faces) |f| {
        for (f) |vi| {
            const p = body_pts[vi];
            try tris.append(allocator, .{
                .x = @floatCast(p.x),
                .y = @floatCast(p.y),
                .z = @floatCast(p.z),
                .r = body_col[0],
                .g = body_col[1],
                .b = body_col[2],
            });
        }
    }

    for (0..4) |idx| {
        const next = (idx + 1) % 4;
        const p_a = corners_w[idx];
        const p_b = corners_w[next];
        try lines.appendSlice(allocator, &.{
            .{ .x = @floatCast(c_pos.x), .y = @floatCast(c_pos.y), .z = @floatCast(c_pos.z), .r = line_col[0], .g = line_col[1], .b = line_col[2] },
            .{ .x = @floatCast(p_a.x), .y = @floatCast(p_a.y), .z = @floatCast(p_a.z), .r = line_col[0], .g = line_col[1], .b = line_col[2] },
            .{ .x = @floatCast(p_a.x), .y = @floatCast(p_a.y), .z = @floatCast(p_a.z), .r = line_col[0], .g = line_col[1], .b = line_col[2] },
            .{ .x = @floatCast(p_b.x), .y = @floatCast(p_b.y), .z = @floatCast(p_b.z), .r = line_col[0], .g = line_col[1], .b = line_col[2] },
        });
    }
}

fn computeCamSpacing(poses: []const CameraPose, registered: []const bool) f64 {
    var sum_d: f64 = 0.0;
    var cnt: usize = 0;
    var prev: ?Vec3 = null;
    for (poses, registered) |p, reg| {
        if (!reg) continue;
        const c_pos = p.cameraCenter();
        if (prev) |pr| {
            sum_d += c_pos.sub(pr).norm();
            cnt += 1;
        }
        prev = c_pos;
    }
    if (cnt == 0) return 0.6;
    return std.math.clamp(sum_d / @as(f64, @floatFromInt(cnt)), 0.15, 3.0);
}

fn drawTexturedQuad(
    vbo: c.GLuint,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    win_w: f32,
    win_h: f32,
    tex: c.GLuint,
    alpha: f32,
    u_tex_loc: c.GLint,
    u_alpha_loc: c.GLint,
) void {
    const x0 = 2.0 * x / win_w - 1.0;
    const x1 = 2.0 * (x + w) / win_w - 1.0;
    const y0 = 1.0 - 2.0 * y / win_h;
    const y1 = 1.0 - 2.0 * (y + h) / win_h;

    const quad = [6]Vertex2DTex{
        .{ .x = x0, .y = y0, .u = 0.0, .v = 0.0 },
        .{ .x = x1, .y = y0, .u = 1.0, .v = 0.0 },
        .{ .x = x1, .y = y1, .u = 1.0, .v = 1.0 },
        .{ .x = x0, .y = y0, .u = 0.0, .v = 0.0 },
        .{ .x = x1, .y = y1, .u = 1.0, .v = 1.0 },
        .{ .x = x0, .y = y1, .u = 0.0, .v = 1.0 },
    };

    c.glActiveTexture(c.GL_TEXTURE0);
    c.glBindTexture(c.GL_TEXTURE_2D, tex);
    c.glUniform1i(u_tex_loc, 0);
    c.glUniform1f(u_alpha_loc, alpha);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
    c.glBufferData(c.GL_ARRAY_BUFFER, @sizeOf(@TypeOf(quad)), &quad, c.GL_DYNAMIC_DRAW);
    c.glDrawArrays(c.GL_TRIANGLES, 0, 6);
}

fn computePhotoFit(win_w: f64, win_h: f64, img_w: f64, img_h: f64, is_cam_view: bool) vm.FittedPhotoRect {
    const pad_top: f64 = if (is_cam_view) 96.0 else 64.0;
    const pad_bot: f64 = 36.0;
    const pad_side: f64 = 16.0;
    return vm.fitImageToMargins(win_w, win_h, img_w, img_h, pad_side, pad_top, pad_side, pad_bot);
}

const PipRect = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
};

fn computePipRect(win_wf: f32, win_hf: f32, img_w: f64, img_h: f64) PipRect {
    const max_pip_w: f32 = @min(240.0, win_wf * 0.22);
    const max_pip_h: f32 = @max(48.0, win_hf * 0.24);
    const aspect: f32 = @floatCast(@max(1.0, img_h) / @max(1.0, img_w));
    const pip_w: f32 = @min(max_pip_w, max_pip_h / @max(0.1, aspect));
    const pip_h: f32 = pip_w * aspect;
    return .{
        .x = win_wf - pip_w - 22.0,
        .y = win_hf - pip_h - 42.0,
        .w = pip_w,
        .h = pip_h,
    };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var ply_path: []const u8 = "output/mesh.ply";
    if (args.len > 1) {
        if (std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h")) {
            std.debug.print("Usage: sfm_viewer [<ply_path>]\n", .{});
            return;
        }
        ply_path = args[1];
    }

    const ply_bytes = Io.Dir.cwd().readFileAlloc(io, ply_path, gpa, .limited(256 * 1024 * 1024)) catch |err| {
        std.log.err("Failed to read PLY file '{s}': {t}", .{ ply_path, err });
        return err;
    };
    var scene = try sfm.io_ply.parsePlyMesh(gpa, ply_bytes);
    gpa.free(ply_bytes);
    defer scene.deinit(gpa);

    const img_w: f64 = @floatFromInt(scene.img_w);
    const img_h: f64 = @floatFromInt(scene.img_h);
    const k = scene.k orelse sfm.mat.Mat3{ .m = .{
        .{ img_w, 0.0, img_w * 0.5 },
        .{ 0.0, img_w, img_h * 0.5 },
        .{ 0.0, 0.0, 1.0 },
    } };

    const cam_count = scene.cameras.len;
    const cam_poses = try gpa.alloc(CameraPose, cam_count);
    defer gpa.free(cam_poses);
    const cam_reg = try gpa.alloc(bool, cam_count);
    defer gpa.free(cam_reg);
    const has_photo = try gpa.alloc(bool, cam_count);
    defer gpa.free(has_photo);

    const cam_textures = try gpa.alloc(c.GLuint, @max(1, cam_count));
    defer gpa.free(cam_textures);

    std.log.info("Loaded '{s}': {d} vertices, {d} triangles, {d} cameras", .{
        ply_path, scene.points.len, scene.faces.len, cam_count,
    });

    if (c.glfwInit() == 0) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 3);
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 3);
    c.glfwWindowHint(c.GLFW_OPENGL_PROFILE, c.GLFW_OPENGL_CORE_PROFILE);
    c.glfwWindowHint(c.GLFW_SAMPLES, 4);

    const window = c.glfwCreateWindow(1280, 800, "3D Reconstruction - Interactive Camera Viewer", null, null) orelse
        return error.WindowCreateFailed;
    defer c.glfwDestroyWindow(window);

    c.glfwMakeContextCurrent(window);
    c.glfwSwapInterval(1);
    _ = c.glfwSetScrollCallback(window, scrollCallback);

    const prog_3d = try linkProgram(vs_3d_src, fs_3d_src);
    defer c.glDeleteProgram(prog_3d);
    const u_mvp_loc = c.glGetUniformLocation(prog_3d, "uMVP");

    const prog_2d_col = try linkProgram(vs_2d_col_src, fs_2d_col_src);
    defer c.glDeleteProgram(prog_2d_col);

    const prog_2d_tex = try linkProgram(vs_2d_tex_src, fs_2d_tex_src);
    defer c.glDeleteProgram(prog_2d_tex);
    const u_tex_loc = c.glGetUniformLocation(prog_2d_tex, "uTex");
    const u_alpha_loc = c.glGetUniformLocation(prog_2d_tex, "uAlpha");

    const mesh_verts = try gpa.alloc(Vertex3D, scene.points.len);
    defer gpa.free(mesh_verts);
    for (scene.points, scene.colors, 0..) |pt, col, idx| {
        mesh_verts[idx] = .{
            .x = @floatCast(pt.x),
            .y = @floatCast(pt.y),
            .z = @floatCast(pt.z),
            .r = @as(f32, @floatFromInt(col.r)) / 255.0,
            .g = @as(f32, @floatFromInt(col.g)) / 255.0,
            .b = @as(f32, @floatFromInt(col.b)) / 255.0,
        };
    }

    const mesh_indices = try gpa.alloc(u32, scene.faces.len * 3);
    defer gpa.free(mesh_indices);
    for (scene.faces, 0..) |f, idx| {
        mesh_indices[idx * 3 + 0] = @intCast(f[0]);
        mesh_indices[idx * 3 + 1] = @intCast(f[1]);
        mesh_indices[idx * 3 + 2] = @intCast(f[2]);
    }

    var mesh_vao: c.GLuint = 0;
    var mesh_vbo: c.GLuint = 0;
    var mesh_ebo: c.GLuint = 0;
    c.glGenVertexArrays(1, &mesh_vao);
    c.glGenBuffers(1, &mesh_vbo);
    c.glGenBuffers(1, &mesh_ebo);
    c.glBindVertexArray(mesh_vao);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, mesh_vbo);
    c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(mesh_verts.len * @sizeOf(Vertex3D)), mesh_verts.ptr, c.GL_STATIC_DRAW);
    c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, mesh_ebo);
    c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, @intCast(mesh_indices.len * @sizeOf(u32)), mesh_indices.ptr, c.GL_STATIC_DRAW);
    c.glEnableVertexAttribArray(0);
    c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex3D), @ptrFromInt(0));
    c.glEnableVertexAttribArray(1);
    c.glVertexAttribPointer(1, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex3D), @ptrFromInt(@offsetOf(Vertex3D, "r")));

    var gizmo_vao: c.GLuint = 0;
    var gizmo_vbo: c.GLuint = 0;
    c.glGenVertexArrays(1, &gizmo_vao);
    c.glGenBuffers(1, &gizmo_vbo);
    c.glBindVertexArray(gizmo_vao);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, gizmo_vbo);
    c.glEnableVertexAttribArray(0);
    c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex3D), @ptrFromInt(0));
    c.glEnableVertexAttribArray(1);
    c.glVertexAttribPointer(1, 3, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex3D), @ptrFromInt(@offsetOf(Vertex3D, "r")));

    var ui_vao: c.GLuint = 0;
    var ui_vbo: c.GLuint = 0;
    c.glGenVertexArrays(1, &ui_vao);
    c.glGenBuffers(1, &ui_vbo);
    c.glBindVertexArray(ui_vao);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, ui_vbo);
    c.glEnableVertexAttribArray(0);
    c.glVertexAttribPointer(0, 2, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex2DColor), @ptrFromInt(0));
    c.glEnableVertexAttribArray(1);
    c.glVertexAttribPointer(1, 4, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex2DColor), @ptrFromInt(@offsetOf(Vertex2DColor, "r")));

    var tex_vao: c.GLuint = 0;
    var tex_vbo: c.GLuint = 0;
    c.glGenVertexArrays(1, &tex_vao);
    c.glGenBuffers(1, &tex_vbo);
    c.glBindVertexArray(tex_vao);
    c.glBindBuffer(c.GL_ARRAY_BUFFER, tex_vbo);
    c.glEnableVertexAttribArray(0);
    c.glVertexAttribPointer(0, 2, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex2DTex), @ptrFromInt(0));
    c.glEnableVertexAttribArray(1);
    c.glVertexAttribPointer(1, 2, c.GL_FLOAT, c.GL_FALSE, @sizeOf(Vertex2DTex), @ptrFromInt(@offsetOf(Vertex2DTex, "u")));

    if (cam_count > 0) {
        c.glGenTextures(@intCast(cam_count), cam_textures.ptr);
    }
    defer if (cam_count > 0) {
        c.glDeleteTextures(@intCast(cam_count), cam_textures.ptr);
    };
    c.glPixelStorei(c.GL_UNPACK_ALIGNMENT, 1);

    for (scene.cameras, 0..) |cam, idx| {
        cam_poses[idx] = cam.pose;
        cam_reg[idx] = true;
        has_photo[idx] = false;

        if (sfm.dataset.loadImage(gpa, cam.image_path)) |loaded_img| {
            var img = loaded_img;
            defer img.deinit(gpa);
            _ = sfm.dataset.downscaleIfNeeded(gpa, &img, @max(scene.img_w, scene.img_h)) catch {};
            c.glBindTexture(c.GL_TEXTURE_2D, cam_textures[idx]);
            c.glTexParameteri(c.GL_TEXTURE_2D, c.GL_TEXTURE_MIN_FILTER, c.GL_LINEAR);
            c.glTexParameteri(c.GL_TEXTURE_2D, c.GL_TEXTURE_MAG_FILTER, c.GL_LINEAR);
            c.glTexParameteri(c.GL_TEXTURE_2D, c.GL_TEXTURE_WRAP_S, c.GL_CLAMP_TO_EDGE);
            c.glTexParameteri(c.GL_TEXTURE_2D, c.GL_TEXTURE_WRAP_T, c.GL_CLAMP_TO_EDGE);
            c.glTexImage2D(
                c.GL_TEXTURE_2D,
                0,
                c.GL_RGB8,
                @intCast(img.width),
                @intCast(img.height),
                0,
                c.GL_RGB,
                c.GL_UNSIGNED_BYTE,
                img.rgb.ptr,
            );
            has_photo[idx] = true;
        } else |_| {}
    }

    const cam_spacing = computeCamSpacing(cam_poses, cam_reg);
    const frustum_depth = cam_spacing * 0.42;
    const hit_radius = cam_spacing * 0.34;

    var ctrl = vm.ViewerController.init(scene.points, cam_poses, cam_spacing);
    var hover_cam: ?usize = null;
    var show_pip: bool = true;

    var gizmo_lines: std.ArrayList(Vertex3D) = .empty;
    defer gizmo_lines.deinit(gpa);
    var gizmo_tris: std.ArrayList(Vertex3D) = .empty;
    defer gizmo_tris.deinit(gpa);
    var ui_batch: Ui2DBatch = .{};
    defer ui_batch.deinit(gpa);
    var ui_buttons: std.ArrayList(vm.UiButton) = .empty;
    defer ui_buttons.deinit(gpa);

    var prev_left = false;
    var prev_right = false;
    var press_mx: f64 = 0;
    var press_my: f64 = 0;
    var last_mx: f64 = 0;
    var last_my: f64 = 0;
    var dragged = false;
    var press_over_ui = false;
    var press_right_over_ui = false;

    var prev_key_f = false;
    var prev_key_r = false;
    var prev_key_p = false;
    var prev_key_left = false;
    var prev_key_right = false;

    var last_time = Io.Clock.now(.awake, io);

    while (c.glfwWindowShouldClose(window) == 0) {
        c.glfwPollEvents();

        const now_time = Io.Clock.now(.awake, io);
        const dt_ns = last_time.durationTo(now_time).nanoseconds;
        last_time = now_time;
        const dt = std.math.clamp(@as(f64, @floatFromInt(dt_ns)) / 1e9, 0.001, 0.1);

        var win_w_i: c_int = 1280;
        var win_h_i: c_int = 800;
        c.glfwGetWindowSize(window, &win_w_i, &win_h_i);
        var fb_w_i: c_int = 1280;
        var fb_h_i: c_int = 800;
        c.glfwGetFramebufferSize(window, &fb_w_i, &fb_h_i);

        const win_w: f64 = @max(1.0, @as(f64, @floatFromInt(win_w_i)));
        const win_h: f64 = @max(1.0, @as(f64, @floatFromInt(win_h_i)));
        const win_wf: f32 = @floatCast(win_w);
        const win_hf: f32 = @floatCast(win_h);

        var mx: f64 = 0;
        var my: f64 = 0;
        c.glfwGetCursorPos(window, &mx, &my);

        // Build HUD button layout
        ui_buttons.clearRetainingCapacity();
        try ui_buttons.append(gpa, .{
            .id = -1,
            .x = 16,
            .y = 14,
            .w = 136,
            .h = 30,
            .label = "FREE ORBIT (F)",
            .active = ctrl.mode == .orbit,
        });

        var bx: f32 = 162;
        for (0..cam_count) |cam_idx| {
            if (!cam_reg[cam_idx]) continue;
            const lbl = try std.fmt.allocPrint(arena, "{d:02}", .{cam_idx});
            try ui_buttons.append(gpa, .{
                .id = @intCast(cam_idx),
                .x = bx,
                .y = 14,
                .w = 34,
                .h = 30,
                .label = lbl,
                .active = ctrl.active_cam == cam_idx,
            });
            bx += 40;
        }

        if (ctrl.active_cam) |ac| {
            if (has_photo[ac]) {
                const ov_lbl: []const u8 = switch (ctrl.overlay_mode) {
                    0 => "OVERLAY: OFF (R)",
                    1 => "OVERLAY: 50% (R)",
                    else => "OVERLAY: 100% (R)",
                };
                try ui_buttons.append(gpa, .{
                    .id = -2,
                    .x = 16,
                    .y = 52,
                    .w = 156,
                    .h = 28,
                    .label = ov_lbl,
                    .active = ctrl.overlay_mode > 0 and !ctrl.cam_modified,
                });
                try ui_buttons.append(gpa, .{
                    .id = -3,
                    .x = 180,
                    .y = 52,
                    .w = 112,
                    .h = 28,
                    .label = if (show_pip) "PIP: ON (P)" else "PIP: OFF (P)",
                    .active = show_pip,
                });
            }
        }

        var over_ui = false;
        for (ui_buttons.items) |btn| {
            if (btn.contains(mx, my)) {
                over_ui = true;
                break;
            }
        }

        const pick_fit = computePhotoFit(win_w, win_h, img_w, img_h, ctrl.mode == .camera);
        const pick_proj = vm.buildProjectionFromKWithFit(k, pick_fit, win_w, win_h, 0.08, 250.0);
        const view = vm.poseToViewMatrix(ctrl.current_pose);
        const viewer_eye = ctrl.current_pose.cameraCenter();

        hover_cam = null;
        if (!over_ui and cam_count > 0) {
            if (vm.screenPointToRay(mx, my, win_w, win_h, pick_proj, view)) |ray| {
                hover_cam = vm.pickCameraExcludingEye(ray, viewer_eye, cam_poses, cam_reg, hit_radius, hit_radius);
            }
        }

        // Keyboard shortcuts
        const key_f = c.glfwGetKey(window, c.GLFW_KEY_F) == c.GLFW_PRESS or
            c.glfwGetKey(window, c.GLFW_KEY_ESCAPE) == c.GLFW_PRESS;
        if (key_f and !prev_key_f) {
            ctrl.enterOrbit();
        }
        prev_key_f = key_f;

        const key_r = c.glfwGetKey(window, c.GLFW_KEY_R) == c.GLFW_PRESS;
        if (key_r and !prev_key_r and ctrl.active_cam != null) {
            ctrl.toggleOverlay(cam_poses);
        }
        prev_key_r = key_r;

        const key_p = c.glfwGetKey(window, c.GLFW_KEY_P) == c.GLFW_PRESS;
        if (key_p and !prev_key_p and ctrl.active_cam != null) {
            show_pip = !show_pip;
        }
        prev_key_p = key_p;

        const key_left = c.glfwGetKey(window, c.GLFW_KEY_LEFT) == c.GLFW_PRESS;
        const key_right = c.glfwGetKey(window, c.GLFW_KEY_RIGHT) == c.GLFW_PRESS;
        if (cam_count > 0 and ((key_left and !prev_key_left) or (key_right and !prev_key_right))) {
            const step: isize = if (key_left and !prev_key_left) -1 else 1;
            const n_imgs: isize = @intCast(cam_count);
            var next_idx: isize = if (ctrl.active_cam) |ac| @as(isize, @intCast(ac)) else if (step > 0) -1 else 0;
            var tries: usize = 0;
            while (tries < cam_count) : (tries += 1) {
                next_idx = @mod(next_idx + step, n_imgs);
                const ui_idx: usize = @intCast(next_idx);
                if (cam_reg[ui_idx]) {
                    ctrl.enterCamera(ui_idx, cam_poses[ui_idx]);
                    break;
                }
            }
        }
        prev_key_left = key_left;
        prev_key_right = key_right;

        // WASD/QE camera-local displacement
        var d_right: f64 = 0.0;
        var d_up: f64 = 0.0;
        var d_fwd: f64 = 0.0;
        if (c.glfwGetKey(window, c.GLFW_KEY_D) == c.GLFW_PRESS) d_right += 1.0;
        if (c.glfwGetKey(window, c.GLFW_KEY_A) == c.GLFW_PRESS) d_right -= 1.0;
        if (c.glfwGetKey(window, c.GLFW_KEY_E) == c.GLFW_PRESS) d_up += 1.0;
        if (c.glfwGetKey(window, c.GLFW_KEY_Q) == c.GLFW_PRESS) d_up -= 1.0;
        if (c.glfwGetKey(window, c.GLFW_KEY_W) == c.GLFW_PRESS) d_fwd += 1.0;
        if (c.glfwGetKey(window, c.GLFW_KEY_S) == c.GLFW_PRESS) d_fwd -= 1.0;
        const move_norm = @sqrt(d_right * d_right + d_up * d_up + d_fwd * d_fwd);
        if (move_norm > 0.0) {
            const move_step = dt * @max(0.5, cam_spacing * 2.5) / move_norm;
            ctrl.moveWasd(d_right * move_step, d_up * move_step, d_fwd * move_step);
        }

        // Mouse input
        const left_down = c.glfwGetMouseButton(window, c.GLFW_MOUSE_BUTTON_LEFT) == c.GLFW_PRESS;
        const right_down = c.glfwGetMouseButton(window, c.GLFW_MOUSE_BUTTON_RIGHT) == c.GLFW_PRESS;
        const shift_down = c.glfwGetKey(window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS or
            c.glfwGetKey(window, c.GLFW_KEY_RIGHT_SHIFT) == c.GLFW_PRESS;

        if (left_down and !prev_left) {
            press_mx = mx;
            press_my = my;
            dragged = false;
            press_over_ui = over_ui;
        }
        if (right_down and !prev_right) {
            press_right_over_ui = over_ui;
        }

        if (left_down and prev_left) {
            const dx = mx - last_mx;
            const dy = my - last_my;
            if (@abs(mx - press_mx) > 5.0 or @abs(my - press_my) > 5.0) {
                dragged = true;
            }
            if (dragged and !press_over_ui) {
                if (shift_down) {
                    ctrl.panDrag(dx / win_w, dy / win_h);
                } else {
                    ctrl.rotateDrag(dx, dy);
                }
            }
        }

        if (right_down and prev_right and !press_right_over_ui) {
            const dx = mx - last_mx;
            const dy = my - last_my;
            if (@abs(dx) > 0.1 or @abs(dy) > 0.1) {
                ctrl.panDrag(dx / win_w, dy / win_h);
            }
        }

        // Handle short click release: UI buttons OR clicking a 3D camera
        if (!left_down and prev_left and !dragged) {
            var handled = false;
            for (ui_buttons.items) |btn| {
                if (btn.contains(mx, my)) {
                    handled = true;
                    if (btn.id == -1) {
                        ctrl.enterOrbit();
                    } else if (btn.id == -2) {
                        ctrl.toggleOverlay(cam_poses);
                    } else if (btn.id == -3) {
                        show_pip = !show_pip;
                    } else if (btn.id >= 0) {
                        const cam_idx: usize = @intCast(btn.id);
                        ctrl.enterCamera(cam_idx, cam_poses[cam_idx]);
                    }
                    break;
                }
            }

            if (!handled) {
                if (hover_cam) |cam_idx| {
                    ctrl.enterCamera(cam_idx, cam_poses[cam_idx]);
                }
            }
        }

        if (g_scroll_y != 0.0) {
            ctrl.scrollMove(g_scroll_y);
            g_scroll_y = 0.0;
        }

        prev_left = left_down;
        prev_right = right_down;
        last_mx = mx;
        last_my = my;

        ctrl.tick(dt);

        // Render frame
        const fit = computePhotoFit(win_w, win_h, img_w, img_h, ctrl.mode == .camera);
        const proj = vm.buildProjectionFromKWithFit(k, fit, win_w, win_h, 0.08, 250.0);

        c.glViewport(0, 0, fb_w_i, fb_h_i);
        c.glClearColor(0.07, 0.08, 0.10, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);

        c.glEnable(c.GL_DEPTH_TEST);
        c.glDisable(c.GL_CULL_FACE);

        const mvp = proj.mul(vm.poseToViewMatrix(ctrl.current_pose));
        c.glUseProgram(prog_3d);
        c.glUniformMatrix4fv(u_mvp_loc, 1, c.GL_FALSE, &mvp.m);

        // Draw reconstructed mesh
        c.glBindVertexArray(mesh_vao);
        c.glDrawElements(c.GL_TRIANGLES, @intCast(mesh_indices.len), c.GL_UNSIGNED_INT, null);

        // Build & draw camera frustums + markers (hide gizmos surrounding the viewer eye)
        gizmo_lines.clearRetainingCapacity();
        gizmo_tris.clearRetainingCapacity();
        const render_eye = ctrl.current_pose.cameraCenter();
        const gizmo_hide_r = frustum_depth * 0.35;
        for (cam_poses, cam_reg, 0..) |p, is_reg, cam_idx| {
            if (!is_reg) continue;
            if (p.cameraCenter().sub(render_eye).norm() < gizmo_hide_r) continue;

            const is_hover = hover_cam == cam_idx;
            const line_col: [3]f32 = if (is_hover)
                .{ 1.0, 0.90, 0.20 }
            else if (ctrl.active_cam == cam_idx)
                .{ 0.20, 0.85, 1.0 }
            else
                .{ 0.20, 0.92, 0.45 };
            const body_col: [3]f32 = if (is_hover)
                .{ 1.0, 0.80, 0.10 }
            else
                .{ 0.12, 0.80, 0.35 };

            try appendCameraGizmoLines(
                gpa,
                &gizmo_lines,
                &gizmo_tris,
                p,
                k,
                img_w,
                img_h,
                frustum_depth,
                line_col,
                body_col,
            );
        }

        c.glBindVertexArray(gizmo_vao);
        c.glBindBuffer(c.GL_ARRAY_BUFFER, gizmo_vbo);
        if (gizmo_tris.items.len > 0) {
            c.glBufferData(
                c.GL_ARRAY_BUFFER,
                @intCast(gizmo_tris.items.len * @sizeOf(Vertex3D)),
                gizmo_tris.items.ptr,
                c.GL_DYNAMIC_DRAW,
            );
            c.glDrawArrays(c.GL_TRIANGLES, 0, @intCast(gizmo_tris.items.len));
        }
        if (gizmo_lines.items.len > 0) {
            c.glLineWidth(1.8);
            c.glBufferData(
                c.GL_ARRAY_BUFFER,
                @intCast(gizmo_lines.items.len * @sizeOf(Vertex3D)),
                gizmo_lines.items.ptr,
                c.GL_DYNAMIC_DRAW,
            );
            c.glDrawArrays(c.GL_LINES, 0, @intCast(gizmo_lines.items.len));
        }

        // 2D Overlay + HUD pass
        c.glDisable(c.GL_DEPTH_TEST);
        c.glEnable(c.GL_BLEND);
        c.glBlendFunc(c.GL_SRC_ALPHA, c.GL_ONE_MINUS_SRC_ALPHA);

        // Optional real photo overlay directly aligned with the active camera projection
        if (ctrl.shouldDrawPhotoOverlay()) {
            const ac = ctrl.active_cam.?;
            if (has_photo[ac]) {
                const alpha: f32 = if (ctrl.overlay_mode == 1) 0.5 else 1.0;
                c.glUseProgram(prog_2d_tex);
                c.glBindVertexArray(tex_vao);
                drawTexturedQuad(tex_vbo, fit.x, fit.y, fit.w, fit.h, win_wf, win_hf, cam_textures[ac], alpha, u_tex_loc, u_alpha_loc);
            }
        }

        // Draw HUD panels & buttons
        ui_batch.clear();
        if (ctrl.mode == .camera and !ctrl.cam_modified) {
            try ui_batch.addRectBorder(
                gpa,
                fit.x - 1.5,
                fit.y - 1.5,
                fit.w + 3.0,
                fit.h + 3.0,
                1.5,
                win_wf,
                win_hf,
                .{ 0.25, 0.78, 1.0, 0.85 },
            );
        }
        try ui_batch.addRect(gpa, 0, 0, win_wf, if (ctrl.mode == .camera) 88 else 56, win_wf, win_hf, .{ 0.05, 0.06, 0.08, 0.78 });

        for (ui_buttons.items) |btn| {
            const hovered = btn.contains(mx, my);
            const bg: [4]f32 = if (btn.active)
                .{ 0.16, 0.56, 0.88, 0.92 }
            else if (hovered)
                .{ 0.28, 0.32, 0.40, 0.92 }
            else
                .{ 0.16, 0.19, 0.24, 0.88 };
            try ui_batch.addRect(gpa, btn.x, btn.y, btn.w, btn.h, win_wf, win_hf, bg);
            try ui_batch.addRectBorder(gpa, btn.x, btn.y, btn.w, btn.h, 1.0, win_wf, win_hf, .{ 0.42, 0.48, 0.58, 0.85 });
            try ui_batch.addText(gpa, btn.x + 8, btn.y + 9, 1.6, btn.label, win_wf, win_hf, .{ 0.96, 0.97, 0.99, 1.0 });
        }

        // Camera label over hovered 3D marker
        if (hover_cam) |hc| {
            const c_w = cam_poses[hc].cameraCenter();
            const clip = mvp.mulVec4(.{ c_w.x, c_w.y, c_w.z, 1.0 });
            if (clip[3] > 0.01) {
                const sx: f32 = @floatCast((clip[0] / clip[3] + 1.0) * 0.5 * win_w);
                const sy: f32 = @floatCast((1.0 - clip[1] / clip[3]) * 0.5 * win_h);
                const tip = try std.fmt.allocPrint(arena, "CAM {d:02} CLICK TO VIEW", .{hc});
                const tip_w: f32 = @as(f32, @floatFromInt(tip.len)) * 6.0 * 1.6 + 16.0;
                try ui_batch.addRect(gpa, sx - tip_w * 0.5, sy - 38, tip_w, 24, win_wf, win_hf, .{ 0.08, 0.10, 0.13, 0.90 });
                try ui_batch.addRectBorder(gpa, sx - tip_w * 0.5, sy - 38, tip_w, 24, 1.0, win_wf, win_hf, .{ 1.0, 0.85, 0.20, 0.95 });
                try ui_batch.addText(gpa, sx - tip_w * 0.5 + 8, sy - 31, 1.6, tip, win_wf, win_hf, .{ 1.0, 0.90, 0.25, 1.0 });
            }
        }

        // Bottom status strip
        const status = if (ctrl.active_cam) |ac|
            if (ctrl.cam_modified)
                try std.fmt.allocPrint(arena, "CAMERA {d:02} FREE MOVE | CLICK {d:02}: RESET | DRAG: LOOK | WASD/SCROLL/R-DRAG: MOVE | F: ORBIT", .{ ac, ac })
            else
                try std.fmt.allocPrint(arena, "CAMERA {d:02} VIEW ({s}) | DRAG: LOOK | WASD/SCROLL/R-DRAG: MOVE | R: OVERLAY | F: ORBIT", .{ ac, std.fs.path.basename(scene.cameras[ac].image_path) })
        else
            "FREE ORBIT | CLICK CAMERA: ENTER FPS VIEW | DRAG: ORBIT | R-DRAG: PAN | WASD/SCROLL: MOVE";
        try ui_batch.addRect(gpa, 0, win_hf - 28, win_wf, 28, win_wf, win_hf, .{ 0.05, 0.06, 0.08, 0.80 });
        try ui_batch.addText(gpa, 14, win_hf - 19, 1.5, status, win_wf, win_hf, .{ 0.85, 0.89, 0.95, 1.0 });

        // PiP card border & label if active camera + show_pip
        const pip_rect: ?PipRect = if (ctrl.active_cam) |ac|
            if (has_photo[ac] and show_pip) computePipRect(win_wf, win_hf, img_w, img_h) else null
        else
            null;

        if (pip_rect) |pip| {
            try ui_batch.addRect(gpa, pip.x - 6, pip.y - 24, pip.w + 12, pip.h + 30, win_wf, win_hf, .{ 0.06, 0.07, 0.09, 0.90 });
            try ui_batch.addRectBorder(gpa, pip.x - 6, pip.y - 24, pip.w + 12, pip.h + 30, 1.5, win_wf, win_hf, .{ 0.25, 0.72, 1.0, 0.90 });
            try ui_batch.addText(gpa, pip.x, pip.y - 17, 1.5, "REAL PHOTO REFERENCE", win_wf, win_hf, .{ 0.55, 0.88, 1.0, 1.0 });
        }

        c.glUseProgram(prog_2d_col);
        c.glBindVertexArray(ui_vao);
        c.glBindBuffer(c.GL_ARRAY_BUFFER, ui_vbo);
        if (ui_batch.verts.items.len > 0) {
            c.glBufferData(
                c.GL_ARRAY_BUFFER,
                @intCast(ui_batch.verts.items.len * @sizeOf(Vertex2DColor)),
                ui_batch.verts.items.ptr,
                c.GL_DYNAMIC_DRAW,
            );
            c.glDrawArrays(c.GL_TRIANGLES, 0, @intCast(ui_batch.verts.items.len));
        }

        // Draw real photo PiP thumbnail inside card
        if (pip_rect) |pip| {
            const ac = ctrl.active_cam.?;
            c.glUseProgram(prog_2d_tex);
            c.glBindVertexArray(tex_vao);
            drawTexturedQuad(tex_vbo, pip.x, pip.y, pip.w, pip.h, win_wf, win_hf, cam_textures[ac], 1.0, u_tex_loc, u_alpha_loc);
        }

        c.glDisable(c.GL_BLEND);
        c.glfwSwapBuffers(window);
    }
}
