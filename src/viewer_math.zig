const std = @import("std");
const Vec3 = @import("math/vec.zig").Vec3;
const Mat3 = @import("math/mat.zig").Mat3;
const lie = @import("math/lie.zig");
const CameraPose = lie.CameraPose;
const render = @import("render.zig");

pub const Mat4 = struct {
    // Column-major for OpenGL (element at col c, row r is m[c * 4 + r])
    m: [16]f32,

    pub const identity: Mat4 = .{ .m = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    } };

    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        var out: [16]f32 = undefined;
        for (0..4) |col| {
            for (0..4) |row| {
                var sum: f32 = 0.0;
                for (0..4) |k| {
                    sum += a.m[k * 4 + row] * b.m[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        return .{ .m = out };
    }

    pub fn mulVec4(a: Mat4, v: [4]f64) [4]f64 {
        var out: [4]f64 = .{ 0, 0, 0, 0 };
        for (0..4) |col| {
            for (0..4) |row| {
                out[row] += @as(f64, a.m[col * 4 + row]) * v[col];
            }
        }
        return out;
    }

    pub fn inverse(mat: Mat4) ?Mat4 {
        var m: [16]f64 = undefined;
        for (0..16) |i| m[i] = mat.m[i];

        var inv: [16]f64 = undefined;
        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
            m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
            m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
            m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
            m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
            m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
            m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
            m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
            m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
            m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
            m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if (@abs(det) < 1e-18) return null;
        const inv_det = 1.0 / det;

        var out: [16]f32 = undefined;
        for (0..16) |i| out[i] = @floatCast(inv[i] * inv_det);
        return .{ .m = out };
    }
};

pub const FittedPhotoRect = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    scale: f64,
};

pub fn fitImageToMargins(
    win_w: f64,
    win_h: f64,
    img_w: f64,
    img_h: f64,
    pad_left: f64,
    pad_top: f64,
    pad_right: f64,
    pad_bot: f64,
) FittedPhotoRect {
    const avail_w = @max(1.0, win_w - pad_left - pad_right);
    const avail_h = @max(1.0, win_h - pad_top - pad_bot);
    const scale = @min(avail_w / @max(1.0, img_w), avail_h / @max(1.0, img_h));
    const drawn_w = img_w * scale;
    const drawn_h = img_h * scale;
    return .{
        .x = @floatCast(pad_left + (avail_w - drawn_w) * 0.5),
        .y = @floatCast(pad_top + (avail_h - drawn_h) * 0.5),
        .w = @floatCast(drawn_w),
        .h = @floatCast(drawn_h),
        .scale = scale,
    };
}

pub fn fitImageToWindow(win_w: f64, win_h: f64, img_w: f64, img_h: f64) FittedPhotoRect {
    return fitImageToMargins(win_w, win_h, img_w, img_h, 0.0, 0.0, 0.0, 0.0);
}

/// Build OpenGL clip-space projection matching SfM camera convention (+X right, +Y down, +Z forward)
/// mapped into `fit` inside window `(win_w, win_h)` while filling the full viewport.
pub fn buildProjectionFromKWithFit(
    k: Mat3,
    fit: FittedPhotoRect,
    win_w: f64,
    win_h: f64,
    near_z: f64,
    far_z: f64,
) Mat4 {
    const fx_w = k.m[0][0] * fit.scale;
    const fy_w = k.m[1][1] * fit.scale;
    const cx_w = @as(f64, fit.x) + k.m[0][2] * fit.scale;
    const cy_w = @as(f64, fit.y) + k.m[1][2] * fit.scale;

    const a: f32 = @floatCast(2.0 * fx_w / win_w);
    const b: f32 = @floatCast(-2.0 * fy_w / win_h);
    const c_x: f32 = @floatCast(2.0 * cx_w / win_w - 1.0);
    const c_y: f32 = @floatCast(1.0 - 2.0 * cy_w / win_h);
    const p_z: f32 = @floatCast((far_z + near_z) / (far_z - near_z));
    const q_z: f32 = @floatCast(-2.0 * far_z * near_z / (far_z - near_z));

    return .{ .m = .{
        a,   0,   0,   0,
        0,   b,   0,   0,
        c_x, c_y, p_z, 1,
        0,   0,   q_z, 0,
    } };
}

/// Build OpenGL clip-space projection matching SfM camera convention (+X right, +Y down, +Z forward)
/// scaled to fit `(img_w, img_h)` inside window `(win_w, win_h)` while filling the full viewport.
pub fn buildProjectionFromK(
    k: Mat3,
    img_w: f64,
    img_h: f64,
    win_w: f64,
    win_h: f64,
    near_z: f64,
    far_z: f64,
) Mat4 {
    const fit = fitImageToWindow(win_w, win_h, img_w, img_h);
    return buildProjectionFromKWithFit(k, fit, win_w, win_h, near_z, far_z);
}

pub fn poseToViewMatrix(pose: CameraPose) Mat4 {
    return .{ .m = .{
        @floatCast(pose.r.m[0][0]), @floatCast(pose.r.m[1][0]), @floatCast(pose.r.m[2][0]), 0,
        @floatCast(pose.r.m[0][1]), @floatCast(pose.r.m[1][1]), @floatCast(pose.r.m[2][1]), 0,
        @floatCast(pose.r.m[0][2]), @floatCast(pose.r.m[1][2]), @floatCast(pose.r.m[2][2]), 0,
        @floatCast(pose.t.x),       @floatCast(pose.t.y),       @floatCast(pose.t.z),       1,
    } };
}

pub const OrbitFrame = struct {
    target: Vec3,
    world_up: Vec3,
    base_fwd: Vec3,
    base_right: Vec3,
    yaw: f64,
    pitch: f64,
    dist: f64,

    pub fn fromScene(points: []const Vec3, poses: []const CameraPose) OrbitFrame {
        var mean_pt = Vec3.zero;
        if (points.len > 0) {
            for (points) |p| mean_pt = mean_pt.add(p);
            mean_pt = mean_pt.scale(1.0 / @as(f64, @floatFromInt(points.len)));
        }

        var avg_up = Vec3.zero;
        var avg_fwd = Vec3.zero;
        var avg_cam = Vec3.zero;
        if (poses.len > 0) {
            for (poses) |p| {
                const rt = p.r.transpose();
                avg_up = avg_up.add(rt.mulVec(Vec3.init(0, -1, 0)));
                avg_fwd = avg_fwd.add(rt.mulVec(Vec3.init(0, 0, 1)));
                avg_cam = avg_cam.add(p.cameraCenter());
            }
            avg_up = avg_up.normalize();
            avg_fwd = avg_fwd.normalize();
            avg_cam = avg_cam.scale(1.0 / @as(f64, @floatFromInt(poses.len)));
        } else {
            avg_up = Vec3.init(0, -1, 0);
            avg_fwd = Vec3.init(0, 0, 1);
            avg_cam = Vec3.init(0, 0, -5);
        }

        // Orthogonalize base_fwd to world_up
        var fwd = avg_fwd.sub(avg_up.scale(avg_fwd.dot(avg_up)));
        if (fwd.norm() < 1e-4) fwd = Vec3.init(0, 0, 1);
        fwd = fwd.normalize();
        const right = fwd.cross(avg_up).normalize();

        const span_dist = if (poses.len > 0) @max(2.0, mean_pt.sub(avg_cam).norm() * 1.15) else 6.0;

        return .{
            .target = mean_pt,
            .world_up = avg_up,
            .base_fwd = fwd,
            .base_right = right,
            .yaw = 0.28,
            .pitch = 0.28,
            .dist = span_dist,
        };
    }

    pub fn toPose(self: OrbitFrame) CameraPose {
        const cy = @cos(self.yaw);
        const sy = @sin(self.yaw);
        const cp = @cos(self.pitch);
        const sp = @sin(self.pitch);

        const fwd_h = self.base_fwd.scale(cy).add(self.base_right.scale(sy)).normalize();
        const right = fwd_h.cross(self.world_up).normalize();
        const cam_fwd = fwd_h.scale(cp).sub(self.world_up.scale(sp)).normalize();
        const cam_down = cam_fwd.cross(right).normalize();

        const r = Mat3{ .m = .{
            .{ right.x, right.y, right.z },
            .{ cam_down.x, cam_down.y, cam_down.z },
            .{ cam_fwd.x, cam_fwd.y, cam_fwd.z },
        } };
        const eye = self.target.sub(cam_fwd.scale(self.dist));
        return poseFromRotationAndEye(r, eye);
    }

    pub fn pan(self: *OrbitFrame, dx_norm: f64, dy_norm: f64) void {
        self.target = displaceInCameraPlane(self.target, self.toPose(), dx_norm, dy_norm, self.dist * 0.8);
    }

    pub fn moveTargetLocal(self: *OrbitFrame, d_right: f64, d_up: f64, d_fwd: f64) void {
        self.target = displaceAlongCameraAxes(self.target, self.toPose(), d_right, d_up, d_fwd);
    }
};

pub const CameraAxes = struct {
    right: Vec3,
    down: Vec3,
    fwd: Vec3,
};

pub fn poseFromRotationAndEye(r: Mat3, eye: Vec3) CameraPose {
    return .{
        .r = r,
        .t = r.mulVec(eye).neg(),
    };
}

pub fn cameraAxes(pose: CameraPose) CameraAxes {
    return .{
        .right = Vec3.init(pose.r.m[0][0], pose.r.m[0][1], pose.r.m[0][2]),
        .down = Vec3.init(pose.r.m[1][0], pose.r.m[1][1], pose.r.m[1][2]),
        .fwd = Vec3.init(pose.r.m[2][0], pose.r.m[2][1], pose.r.m[2][2]),
    };
}

pub fn displaceInCameraPlane(pos: Vec3, pose: CameraPose, dx_norm: f64, dy_norm: f64, scale: f64) Vec3 {
    const ax = cameraAxes(pose);
    return pos
        .sub(ax.right.scale(dx_norm * scale))
        .sub(ax.down.scale(dy_norm * scale));
}

pub fn displaceAlongCameraAxes(pos: Vec3, pose: CameraPose, d_right: f64, d_up: f64, d_fwd: f64) Vec3 {
    const ax = cameraAxes(pose);
    return pos
        .add(ax.right.scale(d_right))
        .sub(ax.down.scale(d_up))
        .add(ax.fwd.scale(d_fwd));
}

/// First-person view anchored to a camera's initial optical center and coordinate frame;
/// yaw/pitch rotate in place while translation follows the current camera axes.
pub const FirstPersonFrame = struct {
    eye: Vec3,
    base_r: Mat3,
    yaw: f64,
    pitch: f64,

    pub fn fromPose(pose: CameraPose) FirstPersonFrame {
        return .{
            .eye = pose.cameraCenter(),
            .base_r = pose.r,
            .yaw = 0.0,
            .pitch = 0.0,
        };
    }

    pub fn toPose(self: FirstPersonFrame) CameraPose {
        const cy = @cos(self.yaw);
        const sy = @sin(self.yaw);
        const cp = @cos(self.pitch);
        const sp = @sin(self.pitch);

        const r_local = Mat3{ .m = .{
            .{ cy, 0.0, -sy },
            .{ -sy * sp, cp, -cy * sp },
            .{ sy * cp, sp, cy * cp },
        } };
        return poseFromRotationAndEye(r_local.mul(self.base_r), self.eye);
    }

    pub fn rotate(self: *FirstPersonFrame, d_yaw: f64, d_pitch: f64) void {
        self.yaw += d_yaw;
        self.pitch = std.math.clamp(self.pitch + d_pitch, -1.45, 1.45);
    }

    pub fn moveLocal(self: *FirstPersonFrame, d_right: f64, d_up: f64, d_fwd: f64) void {
        self.eye = displaceAlongCameraAxes(self.eye, self.toPose(), d_right, d_up, d_fwd);
    }

    pub fn pan(self: *FirstPersonFrame, dx_norm: f64, dy_norm: f64, scale: f64) void {
        self.eye = displaceInCameraPlane(self.eye, self.toPose(), dx_norm, dy_norm, scale);
    }
};

pub const ViewMode = enum {
    orbit,
    camera,
};

/// Unified viewer state switching between scene Orbit and clicked-camera FirstPerson control.
pub const ViewerController = struct {
    mode: ViewMode = .orbit,
    orbit: OrbitFrame,
    fp: FirstPersonFrame,
    scene_center: Vec3,
    cam_spacing: f64,
    active_cam: ?usize = null,
    cam_modified: bool = false,
    overlay_mode: u8 = 0,
    anim_active: bool = false,
    anim_t: f64 = 1.0,
    anim_from_pose: CameraPose,
    anim_to_pose: CameraPose,
    current_pose: CameraPose,

    pub fn init(points: []const Vec3, cam_poses: []const CameraPose, cam_spacing: f64) ViewerController {
        const orbit = OrbitFrame.fromScene(points, cam_poses);
        const start_pose = orbit.toPose();
        return .{
            .orbit = orbit,
            .fp = FirstPersonFrame.fromPose(start_pose),
            .scene_center = orbit.target,
            .cam_spacing = cam_spacing,
            .anim_from_pose = start_pose,
            .anim_to_pose = start_pose,
            .current_pose = start_pose,
        };
    }

    pub fn enterCamera(self: *ViewerController, cam_idx: usize, pose: CameraPose) void {
        self.mode = .camera;
        self.active_cam = cam_idx;
        self.cam_modified = false;
        self.fp = FirstPersonFrame.fromPose(pose);
        self.startAnimTo(self.fp.toPose());
    }

    pub fn enterOrbit(self: *ViewerController) void {
        self.mode = .orbit;
        self.active_cam = null;
        self.cam_modified = false;
        self.overlay_mode = 0;
        self.startAnimTo(self.orbit.toPose());
    }

    pub fn toggleOverlay(self: *ViewerController, cam_poses: []const CameraPose) void {
        const ac = self.active_cam orelse return;
        if (self.mode == .camera and self.cam_modified) {
            self.enterCamera(ac, cam_poses[ac]);
            self.overlay_mode = 1;
            return;
        }
        self.overlay_mode = (self.overlay_mode + 1) % 3;
    }

    pub fn shouldDrawPhotoOverlay(self: *const ViewerController) bool {
        return self.mode == .camera and self.active_cam != null and !self.cam_modified and !self.anim_active and self.overlay_mode > 0;
    }

    fn startAnimTo(self: *ViewerController, target: CameraPose) void {
        self.anim_active = true;
        self.anim_t = 0.0;
        self.anim_from_pose = self.current_pose;
        self.anim_to_pose = target;
    }

    fn prepareDirectControl(self: *ViewerController) void {
        if (!self.anim_active) return;
        self.anim_active = false;
        if (self.mode == .camera) {
            self.fp = FirstPersonFrame.fromPose(self.current_pose);
        }
    }

    fn markCameraModified(self: *ViewerController) void {
        self.cam_modified = true;
        self.overlay_mode = 0;
    }

    fn panScale(self: *const ViewerController) f64 {
        const dist = self.scene_center.sub(self.fp.eye).norm();
        return std.math.clamp(dist * 0.75, @max(0.4, self.cam_spacing * 0.8), 40.0);
    }

    pub fn rotateDrag(self: *ViewerController, dx: f64, dy: f64) void {
        self.prepareDirectControl();
        switch (self.mode) {
            .camera => {
                self.fp.rotate(dx * 0.0028, dy * 0.0028);
                self.markCameraModified();
                self.current_pose = self.fp.toPose();
            },
            .orbit => {
                self.orbit.yaw += dx * 0.006;
                self.orbit.pitch = std.math.clamp(self.orbit.pitch + dy * 0.006, -1.45, 1.45);
                self.current_pose = self.orbit.toPose();
            },
        }
    }

    pub fn panDrag(self: *ViewerController, dx_norm: f64, dy_norm: f64) void {
        self.prepareDirectControl();
        switch (self.mode) {
            .camera => {
                self.fp.pan(dx_norm, dy_norm, self.panScale());
                self.markCameraModified();
                self.current_pose = self.fp.toPose();
            },
            .orbit => {
                self.orbit.pan(dx_norm, dy_norm);
                self.current_pose = self.orbit.toPose();
            },
        }
    }

    pub fn scrollMove(self: *ViewerController, scroll_y: f64) void {
        self.prepareDirectControl();
        switch (self.mode) {
            .camera => {
                const dist = self.scene_center.sub(self.fp.eye).norm();
                const step = scroll_y * @max(self.cam_spacing * 0.25, dist * 0.08);
                self.fp.moveLocal(0.0, 0.0, step);
                self.markCameraModified();
                self.current_pose = self.fp.toPose();
            },
            .orbit => {
                const zoom_factor = std.math.pow(f64, 0.88, scroll_y);
                self.orbit.dist = std.math.clamp(self.orbit.dist * zoom_factor, 0.3, 80.0);
                self.current_pose = self.orbit.toPose();
            },
        }
    }

    pub fn moveWasd(self: *ViewerController, d_right: f64, d_up: f64, d_fwd: f64) void {
        self.prepareDirectControl();
        switch (self.mode) {
            .camera => {
                self.fp.moveLocal(d_right, d_up, d_fwd);
                self.markCameraModified();
                self.current_pose = self.fp.toPose();
            },
            .orbit => {
                self.orbit.moveTargetLocal(d_right, d_up, d_fwd);
                self.current_pose = self.orbit.toPose();
            },
        }
    }

    pub fn tick(self: *ViewerController, dt: f64) void {
        if (self.anim_active) {
            self.anim_t = @min(1.0, self.anim_t + dt / 0.35);
            const ease = 1.0 - std.math.pow(f64, 1.0 - self.anim_t, 3.0);
            self.current_pose = render.interpolatePose(self.anim_from_pose, self.anim_to_pose, ease);
            if (self.anim_t >= 1.0) {
                self.anim_active = false;
                self.current_pose = self.anim_to_pose;
            }
        } else {
            self.current_pose = switch (self.mode) {
                .camera => self.fp.toPose(),
                .orbit => self.orbit.toPose(),
            };
        }
    }
};

pub const Ray = struct {
    origin: Vec3,
    dir: Vec3,
};

pub fn screenPointToRay(
    mx: f64,
    my: f64,
    win_w: f64,
    win_h: f64,
    proj: Mat4,
    view: Mat4,
) ?Ray {
    const inv_vp = proj.mul(view).inverse() orelse return null;
    const x_ndc = (2.0 * mx / win_w) - 1.0;
    const y_ndc = 1.0 - (2.0 * my / win_h);

    const p_near_clip = inv_vp.mulVec4(.{ x_ndc, y_ndc, -1.0, 1.0 });
    const p_far_clip = inv_vp.mulVec4(.{ x_ndc, y_ndc, 1.0, 1.0 });
    if (@abs(p_near_clip[3]) < 1e-12 or @abs(p_far_clip[3]) < 1e-12) return null;

    const p_near = Vec3.init(
        p_near_clip[0] / p_near_clip[3],
        p_near_clip[1] / p_near_clip[3],
        p_near_clip[2] / p_near_clip[3],
    );
    const p_far = Vec3.init(
        p_far_clip[0] / p_far_clip[3],
        p_far_clip[1] / p_far_clip[3],
        p_far_clip[2] / p_far_clip[3],
    );
    const diff = p_far.sub(p_near);
    if (diff.norm() < 1e-12) return null;
    return .{
        .origin = p_near,
        .dir = diff.normalize(),
    };
}

pub fn raySphereIntersect(ray: Ray, center: Vec3, radius: f64) ?f64 {
    const oc = ray.origin.sub(center);
    const b = oc.dot(ray.dir);
    const c = oc.dot(oc) - radius * radius;
    const disc = b * b - c;
    if (disc < 0.0) return null;
    const sq = @sqrt(disc);
    const t0 = -b - sq;
    if (t0 > 0.0) return t0;
    const t1 = -b + sq;
    if (t1 > 0.0) return t1;
    return null;
}

fn pickCameraWithMinEyeDist(
    ray: Ray,
    eye: ?Vec3,
    poses: []const CameraPose,
    registered: []const bool,
    hit_radius: f64,
    min_eye_dist: f64,
) ?usize {
    var best_idx: ?usize = null;
    var best_t: f64 = std.math.inf(f64);
    for (poses, registered, 0..) |pose, is_reg, idx| {
        if (!is_reg) continue;
        const c = pose.cameraCenter();
        if (eye) |e| {
            if (c.sub(e).norm() <= min_eye_dist) continue;
        }
        if (raySphereIntersect(ray, c, hit_radius)) |t| {
            if (t < best_t) {
                best_t = t;
                best_idx = idx;
            }
        }
    }
    return best_idx;
}

pub fn pickCamera(
    ray: Ray,
    poses: []const CameraPose,
    registered: []const bool,
    hit_radius: f64,
) ?usize {
    return pickCameraWithMinEyeDist(ray, null, poses, registered, hit_radius, 0.0);
}

pub fn pickCameraExcludingEye(
    ray: Ray,
    eye: Vec3,
    poses: []const CameraPose,
    registered: []const bool,
    hit_radius: f64,
    min_eye_dist: f64,
) ?usize {
    return pickCameraWithMinEyeDist(ray, eye, poses, registered, hit_radius, min_eye_dist);
}

pub const UiButton = struct {
    id: i32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: []const u8,
    active: bool = false,

    pub fn contains(self: UiButton, mx: f64, my: f64) bool {
        return mx >= self.x and mx <= self.x + self.w and my >= self.y and my <= self.y + self.h;
    }
};

/// Minimal 5x7 bitmap glyph rows (5 bits per row, MSB = left pixel) for crisp HUD text
pub fn glyph5x7(ch: u8) [7]u8 {
    return switch (ch) {
        '0' => .{ 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E },
        '1' => .{ 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E },
        '2' => .{ 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F },
        '3' => .{ 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E },
        '4' => .{ 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 },
        '5' => .{ 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E },
        '6' => .{ 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E },
        '7' => .{ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 },
        '8' => .{ 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E },
        '9' => .{ 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C },
        'A', 'a' => .{ 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 },
        'B', 'b' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E },
        'C', 'c' => .{ 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E },
        'D', 'd' => .{ 0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C },
        'E', 'e' => .{ 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F },
        'F', 'f' => .{ 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10 },
        'G', 'g' => .{ 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F },
        'H', 'h' => .{ 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 },
        'I', 'i' => .{ 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E },
        'J', 'j' => .{ 0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C },
        'K', 'k' => .{ 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 },
        'L', 'l' => .{ 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F },
        'M', 'm' => .{ 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11 },
        'N', 'n' => .{ 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11 },
        'O', 'o' => .{ 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
        'P', 'p' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 },
        'Q', 'q' => .{ 0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D },
        'R', 'r' => .{ 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 },
        'S', 's' => .{ 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E },
        'T', 't' => .{ 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
        'U', 'u' => .{ 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },
        'V', 'v' => .{ 0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04 },
        'W', 'w' => .{ 0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11 },
        'X', 'x' => .{ 0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11 },
        'Y', 'y' => .{ 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04 },
        'Z', 'z' => .{ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F },
        ':' => .{ 0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00 },
        '.' => .{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C },
        '-' => .{ 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00 },
        '+' => .{ 0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00 },
        '/' => .{ 0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10 },
        '%' => .{ 0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03 },
        '(' => .{ 0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02 },
        ')' => .{ 0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08 },
        '[' => .{ 0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E },
        ']' => .{ 0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E },
        '|' => .{ 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },
        else => .{ 0, 0, 0, 0, 0, 0, 0 },
    };
}

test "buildProjectionFromK and poseToViewMatrix match lie.projectPoint" {
    const k = Mat3{ .m = .{
        .{ 1000.0, 0.0, 600.0 },
        .{ 0.0, 1000.0, 400.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    const r = lie.so3Exp(Vec3.init(0.1, -0.2, 0.05));
    const pose = CameraPose{ .r = r, .t = Vec3.init(0.3, -0.1, 2.0) };
    const pt = Vec3.init(0.4, -0.2, 4.5);

    const uv_ref = lie.projectPoint(k, pose, pt).?;

    const proj = buildProjectionFromK(k, 1200.0, 800.0, 1200.0, 800.0, 0.1, 100.0);
    const view = poseToViewMatrix(pose);
    const clip = proj.mul(view).mulVec4(.{ pt.x, pt.y, pt.z, 1.0 });

    const x_ndc = clip[0] / clip[3];
    const y_ndc = clip[1] / clip[3];
    const u_gl = (x_ndc + 1.0) * 0.5 * 1200.0;
    const v_gl = (1.0 - y_ndc) * 0.5 * 800.0;

    try std.testing.expectApproxEqAbs(uv_ref.x, u_gl, 1e-3);
    try std.testing.expectApproxEqAbs(uv_ref.y, v_gl, 1e-3);
}

test "screenPointToRay and pickCamera select the clicked camera" {
    const k = Mat3{ .m = .{
        .{ 800.0, 0.0, 400.0 },
        .{ 0.0, 800.0, 300.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    const cam_a = CameraPose{ .r = Mat3.identity, .t = Vec3.init(2.0, 0.0, -5.0) }; // center (-2, 0, 5)
    const cam_b = CameraPose{ .r = Mat3.identity, .t = Vec3.init(-2.0, 0.0, -5.0) }; // center (+2, 0, 5)
    const poses = [_]CameraPose{ cam_a, cam_b };
    const reg = [_]bool{ true, true };

    const viewer_pose = CameraPose.identity;
    const proj = buildProjectionFromK(k, 800.0, 600.0, 800.0, 600.0, 0.1, 100.0);
    const view = poseToViewMatrix(viewer_pose);

    const uv_b = lie.projectPoint(k, viewer_pose, cam_b.cameraCenter()).?;
    const ray = screenPointToRay(uv_b.x, uv_b.y, 800.0, 600.0, proj, view).?;
    const hit = pickCamera(ray, &poses, &reg, 0.35);
    try std.testing.expectEqual(@as(?usize, 1), hit);
}

test "OrbitFrame keeps camera looking at target" {
    const pts = [_]Vec3{Vec3.init(0, 0, 5)};
    const poses = [_]CameraPose{CameraPose.identity};
    const orbit = OrbitFrame.fromScene(&pts, &poses);
    const p = orbit.toPose();
    const xc = p.transformPoint(orbit.target);
    try std.testing.expectApproxEqAbs(0.0, xc.x, 1e-5);
    try std.testing.expectApproxEqAbs(0.0, xc.y, 1e-5);
    try std.testing.expectApproxEqAbs(orbit.dist, xc.z, 1e-5);
}

test "fitImageToMargins and buildProjectionFromKWithFit frame exact image bounds" {
    const win_w = 1280.0;
    const win_h = 800.0;
    const pad_left = 16.0;
    const pad_top = 96.0;
    const pad_right = 16.0;
    const pad_bot = 36.0;

    const cases = [_][2]f64{
        .{ 1600.0, 1200.0 },
        .{ 1920.0, 1080.0 },
        .{ 900.0, 1200.0 },
    };

    const r = lie.so3Exp(Vec3.init(-0.15, 0.25, -0.08));
    const pose = CameraPose{ .r = r, .t = Vec3.init(-0.4, 0.2, 3.5) };
    const view = poseToViewMatrix(pose);
    const rt = pose.r.transpose();
    const cam_center = pose.cameraCenter();

    for (cases) |dims| {
        const img_w = dims[0];
        const img_h = dims[1];
        const fit = fitImageToMargins(win_w, win_h, img_w, img_h, pad_left, pad_top, pad_right, pad_bot);

        try std.testing.expect(fit.x >= pad_left);
        try std.testing.expect(fit.y >= pad_top);
        try std.testing.expect(@as(f64, fit.x) + @as(f64, fit.w) <= win_w - pad_right + 1e-4);
        try std.testing.expect(@as(f64, fit.y) + @as(f64, fit.h) <= win_h - pad_bot + 1e-4);

        const fx = img_w * 0.9;
        const fy = img_w * 0.9;
        const cx = img_w * 0.52;
        const cy = img_h * 0.48;
        const k = Mat3{ .m = .{
            .{ fx, 0.0, cx },
            .{ 0.0, fy, cy },
            .{ 0.0, 0.0, 1.0 },
        } };
        const proj = buildProjectionFromKWithFit(k, fit, win_w, win_h, 0.1, 100.0);
        const vp = proj.mul(view);

        const z = 5.0;
        const p_tl_cam = Vec3.init((0.0 - cx) / fx * z, (0.0 - cy) / fy * z, z);
        const p_br_cam = Vec3.init((img_w - cx) / fx * z, (img_h - cy) / fy * z, z);
        const p_tl = cam_center.add(rt.mulVec(p_tl_cam));
        const p_br = cam_center.add(rt.mulVec(p_br_cam));

        const clip_tl = vp.mulVec4(.{ p_tl.x, p_tl.y, p_tl.z, 1.0 });
        const sx_tl = (clip_tl[0] / clip_tl[3] + 1.0) * 0.5 * win_w;
        const sy_tl = (1.0 - clip_tl[1] / clip_tl[3]) * 0.5 * win_h;
        try std.testing.expectApproxEqAbs(@as(f64, fit.x), sx_tl, 1e-3);
        try std.testing.expectApproxEqAbs(@as(f64, fit.y), sy_tl, 1e-3);

        const clip_br = vp.mulVec4(.{ p_br.x, p_br.y, p_br.z, 1.0 });
        const sx_br = (clip_br[0] / clip_br[3] + 1.0) * 0.5 * win_w;
        const sy_br = (1.0 - clip_br[1] / clip_br[3]) * 0.5 * win_h;
        try std.testing.expectApproxEqAbs(@as(f64, fit.x) + @as(f64, fit.w), sx_br, 1e-3);
        try std.testing.expectApproxEqAbs(@as(f64, fit.y) + @as(f64, fit.h), sy_br, 1e-3);
    }
}

test "FirstPersonFrame matches clicked camera pose, rotates in place, and moves along rotated axes" {
    const r0 = lie.so3Exp(Vec3.init(0.18, -0.45, 0.12));
    const cam0 = CameraPose{ .r = r0, .t = Vec3.init(1.2, -0.7, 3.4) };
    var fp = FirstPersonFrame.fromPose(cam0);

    const p0 = fp.toPose();
    for (0..3) |ri| {
        for (0..3) |ci| {
            try std.testing.expectApproxEqAbs(cam0.r.m[ri][ci], p0.r.m[ri][ci], 1e-12);
        }
    }
    const c0 = cam0.cameraCenter();
    const pc0 = p0.cameraCenter();
    try std.testing.expectApproxEqAbs(c0.x, pc0.x, 1e-12);
    try std.testing.expectApproxEqAbs(c0.y, pc0.y, 1e-12);
    try std.testing.expectApproxEqAbs(c0.z, pc0.z, 1e-12);

    // Pure rotation keeps optical center fixed while rotating forward toward camera right/down
    const ax0 = cameraAxes(cam0);
    fp.rotate(std.math.pi * 0.5, 0.0);
    const p_rot = fp.toPose();
    const pc_rot = p_rot.cameraCenter();
    try std.testing.expectApproxEqAbs(c0.x, pc_rot.x, 1e-10);
    try std.testing.expectApproxEqAbs(c0.y, pc_rot.y, 1e-10);
    try std.testing.expectApproxEqAbs(c0.z, pc_rot.z, 1e-10);

    const ax_rot = cameraAxes(p_rot);
    try std.testing.expectApproxEqAbs(ax0.right.x, ax_rot.fwd.x, 1e-10);
    try std.testing.expectApproxEqAbs(ax0.right.y, ax_rot.fwd.y, 1e-10);
    try std.testing.expectApproxEqAbs(ax0.right.z, ax_rot.fwd.z, 1e-10);
    try std.testing.expectApproxEqAbs(1.0, p_rot.r.det(), 1e-10);

    // Displacement moves along the current rotated frame rather than base frame
    fp.moveLocal(0.0, 0.0, 2.0);
    const expected_eye = c0.add(ax0.right.scale(2.0));
    const pc_moved = fp.toPose().cameraCenter();
    try std.testing.expectApproxEqAbs(expected_eye.x, pc_moved.x, 1e-10);
    try std.testing.expectApproxEqAbs(expected_eye.y, pc_moved.y, 1e-10);
    try std.testing.expectApproxEqAbs(expected_eye.z, pc_moved.z, 1e-10);
}

test "pickCameraExcludingEye ignores the occupied camera and selects cameras ahead" {
    const k = Mat3{ .m = .{
        .{ 800.0, 0.0, 400.0 },
        .{ 0.0, 800.0, 300.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    const cam_self = CameraPose.identity;
    const cam_ahead = CameraPose{ .r = Mat3.identity, .t = Vec3.init(0.0, 0.0, -4.0) };
    const poses = [_]CameraPose{ cam_self, cam_ahead };
    const reg = [_]bool{ true, true };

    const proj = buildProjectionFromK(k, 800.0, 600.0, 800.0, 600.0, 0.08, 100.0);
    const view = poseToViewMatrix(cam_self);
    const ray_center = screenPointToRay(400.0, 300.0, 800.0, 600.0, proj, view).?;
    const ray_sky = screenPointToRay(10.0, 10.0, 800.0, 600.0, proj, view).?;

    // Standard pickCamera would self-hit cam_self at t1 = hit_radius - near_z
    try std.testing.expectEqual(@as(?usize, 1), pickCameraExcludingEye(ray_center, cam_self.cameraCenter(), &poses, &reg, 0.35, 0.35));
    try std.testing.expectEqual(@as(?usize, null), pickCameraExcludingEye(ray_sky, cam_self.cameraCenter(), &poses, &reg, 0.35, 0.35));
}

test "ViewerController switches to camera FirstPerson model and rotates/displaces without snapping back" {
    const pts = [_]Vec3{Vec3.init(0, 0, 5)};
    const cam0 = CameraPose{ .r = Mat3.identity, .t = Vec3.init(0.0, 0.0, 0.0) };
    const cam1 = CameraPose{
        .r = lie.so3Exp(Vec3.init(0.05, -0.5, 0.1)),
        .t = Vec3.init(-2.0, 0.3, -1.5),
    };
    const cams = [_]CameraPose{ cam0, cam1 };
    var ctrl = ViewerController.init(&pts, &cams, 0.8);
    try std.testing.expectEqual(ViewMode.orbit, ctrl.mode);
    const orbit_pose0 = ctrl.current_pose;

    ctrl.enterCamera(1, cam1);
    try std.testing.expectEqual(ViewMode.camera, ctrl.mode);
    try std.testing.expectEqual(@as(?usize, 1), ctrl.active_cam);
    ctrl.tick(0.35);

    const c1 = cam1.cameraCenter();
    const landed = ctrl.current_pose.cameraCenter();
    try std.testing.expectApproxEqAbs(c1.x, landed.x, 1e-9);
    try std.testing.expectApproxEqAbs(c1.y, landed.y, 1e-9);
    try std.testing.expectApproxEqAbs(c1.z, landed.z, 1e-9);

    // Drag-rotating after clicking cam1 stays in camera mode (no snap back to orbit_pose0) and rotates in place
    ctrl.overlay_mode = 2;
    ctrl.rotateDrag(120.0, -30.0);
    try std.testing.expectEqual(ViewMode.camera, ctrl.mode);
    try std.testing.expect(ctrl.cam_modified);
    try std.testing.expectEqual(@as(u8, 0), ctrl.overlay_mode);
    const after_rot = ctrl.current_pose;
    try std.testing.expectApproxEqAbs(c1.x, after_rot.cameraCenter().x, 1e-9);
    try std.testing.expectApproxEqAbs(c1.y, after_rot.cameraCenter().y, 1e-9);
    try std.testing.expectApproxEqAbs(c1.z, after_rot.cameraCenter().z, 1e-9);
    try std.testing.expect(after_rot.cameraCenter().sub(orbit_pose0.cameraCenter()).norm() > 0.5);

    // WASD forward moves strictly along the newly rotated forward axis of cam1
    const rotated_ax = cameraAxes(after_rot);
    ctrl.moveWasd(0.0, 0.0, 1.5);
    const moved_c = ctrl.current_pose.cameraCenter();
    const expected_c = c1.add(rotated_ax.fwd.scale(1.5));
    try std.testing.expectApproxEqAbs(expected_c.x, moved_c.x, 1e-9);
    try std.testing.expectApproxEqAbs(expected_c.y, moved_c.y, 1e-9);
    try std.testing.expectApproxEqAbs(expected_c.z, moved_c.z, 1e-9);

    // Returning to orbit restores orbit mode
    ctrl.enterOrbit();
    try std.testing.expectEqual(ViewMode.orbit, ctrl.mode);
    try std.testing.expectEqual(@as(?usize, null), ctrl.active_cam);
}
