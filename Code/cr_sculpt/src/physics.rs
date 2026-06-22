use std::collections::HashMap;
use glam::{FloatExt, Vec3, Vec4, Vec3Swizzles, Vec4Swizzles};

use crate::math;

// ===== 物理碰撞接触点 =====
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub normal: glam::Vec3,        // 碰撞法线（指向推开 A 的方向）
    pub depth: f32,                 // 穿透深度
    pub contact_point: glam::Vec3,   // 世界空间下的碰撞点位置
}

/// 升级版接触点数据，携带对方物体的标识
pub struct MeshContact {
    pub contact: Contact,
    pub other_idx: usize,
}

/// 每个实例的物理状态
#[derive(Clone, Copy)]
pub struct InstancePhysics {
    pub gravity_enabled: bool,
    pub velocity: glam::Vec3,
    pub angular_velocity: glam::Vec3,
    pub sleeping: bool,
}

impl Default for InstancePhysics {
    fn default() -> Self {
        Self {
            gravity_enabled: false,
            velocity: glam::Vec3::ZERO,
            angular_velocity: glam::Vec3::ZERO,
            sleeping: false,
        }
    }
}

/// 模型碰撞体：包含采样点、AABB 包围盒、SDF Grid
pub struct ModelCollider {
    pub sample_points: Vec<glam::Vec3>,
    pub bounds_min: glam::Vec3,
    pub bounds_max: glam::Vec3,
    pub resolution: u32,
    pub sdf: Vec<f32>,
}

// ===================================================================
//  核心物理函数
// ===================================================================

/// 在 SDF Grid 中采样距离与法线
pub fn sample_collider_sdf(collider: &ModelCollider, local_pos: glam::Vec3) -> Option<(f32, glam::Vec3)> {
    let size = collider.bounds_max - collider.bounds_min;
    if size.length_squared() < 1e-10 {
        return None;
    }
    let uvw = (local_pos - collider.bounds_min) / size;
    if uvw.min_element() < 0.0 || uvw.max_element() > 1.0 {
        let clamped = uvw.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
        let outside = (uvw - clamped) * size;
        let normal = outside.normalize_or_zero();
        return Some((outside.length().max(0.01), if normal.length_squared() > 1e-8 { normal } else { glam::Vec3::Y }));
    }

    let r = collider.resolution as i32;
    let grid = uvw * (collider.resolution - 1) as f32;
    let base = grid.floor().as_ivec3().clamp(glam::IVec3::ZERO, glam::IVec3::splat(r - 2));
    let f = grid - base.as_vec3();
    let idx_fn = |x: i32, y: i32, z: i32| -> usize {
        (z as u32 * collider.resolution * collider.resolution + y as u32 * collider.resolution + x as u32) as usize
    };
    let s = |dx: i32, dy: i32, dz: i32| -> f32 {
        collider.sdf[idx_fn(base.x + dx, base.y + dy, base.z + dz)]
    };
    let c00 = s(0, 0, 0).lerp(s(1, 0, 0), f.x);
    let c10 = s(0, 1, 0).lerp(s(1, 1, 0), f.x);
    let c01 = s(0, 0, 1).lerp(s(1, 0, 1), f.x);
    let c11 = s(0, 1, 1).lerp(s(1, 1, 1), f.x);
    let c0 = c00.lerp(c10, f.y);
    let c1 = c01.lerp(c11, f.y);
    let distance = c0.lerp(c1, f.z);

    let step = size / (collider.resolution - 1) as f32;
    let sample_offset = |offset: glam::Vec3| -> f32 {
        let p = local_pos + offset;
        let uvw = ((p - collider.bounds_min) / size).clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
        let grid = uvw * (collider.resolution - 1) as f32;
        let base = grid.floor().as_ivec3().clamp(glam::IVec3::ZERO, glam::IVec3::splat(r - 2));
        let f = grid - base.as_vec3();
        let s = |dx: i32, dy: i32, dz: i32| -> f32 { collider.sdf[idx_fn(base.x + dx, base.y + dy, base.z + dz)] };
        let c00 = s(0, 0, 0).lerp(s(1, 0, 0), f.x);
        let c10 = s(0, 1, 0).lerp(s(1, 1, 0), f.x);
        let c01 = s(0, 0, 1).lerp(s(1, 0, 1), f.x);
        let c11 = s(0, 1, 1).lerp(s(1, 1, 1), f.x);
        c00.lerp(c10, f.y).lerp(c01.lerp(c11, f.y), f.z)
    };
    let normal = glam::Vec3::new(
        sample_offset(glam::Vec3::X * step.x) - sample_offset(-glam::Vec3::X * step.x),
        sample_offset(glam::Vec3::Y * step.y) - sample_offset(-glam::Vec3::Y * step.y),
        sample_offset(glam::Vec3::Z * step.z) - sample_offset(-glam::Vec3::Z * step.z),
    ).normalize_or_zero();
    Some((distance, if normal.length_squared() > 1e-8 { normal } else { glam::Vec3::Y }))
}

/// 同步物理状态数组长度与实例数组一致
pub fn sync_instance_physics_len(
    instances: &[math::InstanceData],
    instance_physics: &mut Vec<InstancePhysics>,
) {
    instance_physics.resize(instances.len(), InstancePhysics::default());
}

/// 收集物体与地面的少量较深碰撞接触点
pub fn detect_ground_collisions(
    instances: &[math::InstanceData],
    model_colliders: &HashMap<u32, ModelCollider>,
    idx: usize,
) -> Vec<Contact> {
    let mut contacts = Vec::new();
    let Some(instance) = instances.get(idx) else { return contacts; };
    let Some(collider) = model_colliders.get(&instance.model_id) else { return contacts; };
    let model_mat = glam::Mat4::from_cols_array_2d(&instance.model_matrix);

    for sample in collider.sample_points.iter().step_by(4) {
        let world_pos = model_mat.transform_point3(*sample);
        if world_pos.y < 0.0 {
            contacts.push(Contact {
                normal: glam::Vec3::Y,
                depth: -world_pos.y,
                contact_point: world_pos,
            });
        }
    }
    contacts.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    contacts.truncate(4);
    contacts
}

/// 双向对称碰撞探测：每个对象对最多返回少量最深接触点
pub fn detect_mesh_collisions_symmetric_detailed(
    instances: &[math::InstanceData],
    model_colliders: &HashMap<u32, ModelCollider>,
    idx: usize,
) -> Vec<MeshContact> {
    let mut contacts = Vec::new();
    let Some(instance_a) = instances.get(idx) else { return contacts; };
    let Some(collider_a) = model_colliders.get(&instance_a.model_id) else { return contacts; };
    let mat_a = glam::Mat4::from_cols_array_2d(&instance_a.model_matrix);
    let mat_a_inv = mat_a.inverse();
    let mat_a_normal = mat_a_inv.transpose();
    let (scale_a, _, center_a) = mat_a.to_scale_rotation_translation();
    let radius_a = (collider_a.bounds_max - collider_a.bounds_min).length() * scale_a.abs().max_element() * 0.5;

    for (other_idx, instance_b) in instances.iter().enumerate() {
        if other_idx == idx { continue; }
        let Some(collider_b) = model_colliders.get(&instance_b.model_id) else { continue; };
        let mat_b = glam::Mat4::from_cols_array_2d(&instance_b.model_matrix);
        let mat_b_inv = mat_b.inverse();
        let mat_b_normal = mat_b_inv.transpose();
        let (scale_b, _, center_b) = mat_b.to_scale_rotation_translation();
        let radius_b = (collider_b.bounds_max - collider_b.bounds_min).length() * scale_b.abs().max_element() * 0.5;

        if center_a.distance(center_b) > radius_a + radius_b + 0.1 {
            continue;
        }

        let step_a = (collider_a.sample_points.len() / 64).max(1);
        let step_b = (collider_b.sample_points.len() / 64).max(1);
        let mut best_ab: Option<MeshContact> = None;
        let mut best_ba: Option<MeshContact> = None;

        for sample_a in collider_a.sample_points.iter().step_by(step_a) {
            let world_pos = mat_a.transform_point3(*sample_a);
            let local_pos = mat_b_inv.transform_point3(world_pos);
            let Some((dist, local_n)) = sample_collider_sdf(collider_b, local_pos) else { continue; };
            if dist >= 0.0 { continue; }
            let mut world_n = mat_b_normal.transform_vector3(local_n).normalize_or_zero();
            if world_n.dot(world_pos - center_b) < 0.0 { world_n = -world_n; }
            if world_n.length_squared() < 1e-8 { world_n = (world_pos - center_b).normalize_or_zero(); }
            if world_n.length_squared() < 1e-8 { continue; }
            let contact = MeshContact {
                contact: Contact { normal: world_n, depth: -dist, contact_point: world_pos },
                other_idx,
            };
            if best_ab.as_ref().map_or(true, |best| contact.contact.depth > best.contact.depth) {
                best_ab = Some(contact);
            }
        }

        for sample_b in collider_b.sample_points.iter().step_by(step_b) {
            let world_pos = mat_b.transform_point3(*sample_b);
            let local_pos = mat_a_inv.transform_point3(world_pos);
            let Some((dist, local_n)) = sample_collider_sdf(collider_a, local_pos) else { continue; };
            if dist >= 0.0 { continue; }
            let mut world_n = mat_a_normal.transform_vector3(local_n).normalize_or_zero();
            if world_n.dot(world_pos - center_a) < 0.0 { world_n = -world_n; }
            if world_n.length_squared() < 1e-8 { world_n = (world_pos - center_a).normalize_or_zero(); }
            if world_n.length_squared() < 1e-8 { continue; }
            let contact = MeshContact {
                contact: Contact { normal: -world_n, depth: -dist, contact_point: world_pos },
                other_idx,
            };
            if best_ba.as_ref().map_or(true, |best| contact.contact.depth > best.contact.depth) {
                best_ba = Some(contact);
            }
        }

        if let Some(contact) = best_ab { contacts.push(contact); }
        if let Some(contact) = best_ba { contacts.push(contact); }
    }
    contacts
}

/// 刚体碰撞求解核心（双体统一求解器）
pub fn resolve_rigid_body_contact_two_bodies(
    pos_a: &mut glam::Vec3,
    vel_a: &mut glam::Vec3,
    ang_a: &mut glam::Vec3,
    has_physics_a: bool,
    pos_b: &mut glam::Vec3,
    vel_b: &mut glam::Vec3,
    ang_b: &mut glam::Vec3,
    has_physics_b: bool,
    contact: Contact,
    restitution: f32,
    friction: f32,
) {
    let inv_mass_a = if has_physics_a { 1.0 } else { 0.0 };
    let inv_mass_b = if has_physics_b { 1.0 } else { 0.0 };
    let inv_inertia_a = if has_physics_a { 1.0 / (0.4 * 1.0 * 0.6f32.powi(2)) } else { 0.0 };
    let inv_inertia_b = if has_physics_b { 1.0 / (0.4 * 1.0 * 0.6f32.powi(2)) } else { 0.0 };

    let total_inv_mass = inv_mass_a + inv_mass_b;
    if total_inv_mass == 0.0 { return; }

    let r_a = contact.contact_point - *pos_a;
    let r_b = contact.contact_point - *pos_b;

    // === 1. 安全位置修正 (Linear Projection) ===
    let slop = 0.005;
    let depth_to_resolve = (contact.depth - slop).max(0.0);
    let penetration_resolve = depth_to_resolve * 0.5;

    if penetration_resolve > 0.0 {
        let ratio_a = inv_mass_a / total_inv_mass;
        let ratio_b = inv_mass_b / total_inv_mass;
        *pos_a += contact.normal * (penetration_resolve * ratio_a);
        *pos_b -= contact.normal * (penetration_resolve * ratio_b);
    }

    // === 2. 法向碰撞冲量 (Normal Impulse) ===
    let v_contact_a = *vel_a + ang_a.cross(r_a);
    let v_contact_b = *vel_b + ang_b.cross(r_b);
    let v_rel = v_contact_a - v_contact_b;

    let vn = v_rel.dot(contact.normal);
    if vn >= 0.0 { return; }

    let active_restitution = if vn.abs() < 0.35 { 0.0 } else { restitution };

    let r_cross_n_a = r_a.cross(contact.normal);
    let r_cross_n_b = r_b.cross(contact.normal);
    let rot_term_a = (r_cross_n_a * inv_inertia_a).cross(r_a).dot(contact.normal);
    let rot_term_b = (r_cross_n_b * inv_inertia_b).cross(r_b).dot(contact.normal);

    let denom_n = inv_mass_a + inv_mass_b + rot_term_a + rot_term_b;
    if denom_n == 0.0 { return; }

    let max_normal_impulse = 6.0;
    let j_n = (-(1.0 + active_restitution) * vn / denom_n).clamp(0.0, max_normal_impulse);

    *vel_a += (j_n * inv_mass_a) * contact.normal;
    *ang_a += r_a.cross(contact.normal * j_n) * inv_inertia_a;
    *vel_b -= (j_n * inv_mass_b) * contact.normal;
    *ang_b -= r_b.cross(contact.normal * j_n) * inv_inertia_b;

    // === 3. 切向摩擦冲量 (Friction Impulse) ===
    let v_contact_a_new = *vel_a + ang_a.cross(r_a);
    let v_contact_b_new = *vel_b + ang_b.cross(r_b);
    let v_rel_new = v_contact_a_new - v_contact_b_new;

    let vt_vec = v_rel_new - (v_rel_new.dot(contact.normal) * contact.normal);
    let vt_len = vt_vec.length();

    if vt_len > 1e-4 {
        let tangent = vt_vec / vt_len;
        let r_cross_t_a = r_a.cross(tangent);
        let r_cross_t_b = r_b.cross(tangent);
        let rot_term_t_a = (r_cross_t_a * inv_inertia_a).cross(r_a).dot(tangent);
        let rot_term_t_b = (r_cross_t_b * inv_inertia_b).cross(r_b).dot(tangent);

        let denom_t = inv_mass_a + inv_mass_b + rot_term_t_a + rot_term_t_b;
        if denom_t > 0.0 {
            let mut j_t = -vt_len / denom_t;
            let max_friction = friction * j_n;
            j_t = j_t.clamp(-max_friction, max_friction);

            *vel_a += (j_t * inv_mass_a) * tangent;
            *ang_a += r_a.cross(tangent * j_t) * inv_inertia_a;
            *vel_b -= (j_t * inv_mass_b) * tangent;
            *ang_b -= r_b.cross(tangent * j_t) * inv_inertia_b;
        }
    }
}

/// 地面接触旋转解算：包含重力支撑力矩、法向冲量、切向摩擦
pub fn resolve_ground_contact_with_rotation(
    pos: glam::Vec3,
    vel: &mut glam::Vec3,
    ang: &mut glam::Vec3,
    contact: Contact,
    dt: f32,
) {
    let inv_mass = 1.0;
    let inv_inertia = 1.0 / (0.4 * 1.0 * 0.6f32.powi(2));
    let normal = glam::Vec3::Y;
    let r = contact.contact_point - pos;
    if r.length_squared() < 1e-8 {
        return;
    }

    // 重力支撑力矩：物体斜着压在支撑点上产生持续旋转恢复
    let support_force = glam::Vec3::Y * 9.8;
    let gravity_torque = r.cross(support_force);
    *ang += gravity_torque * inv_inertia * dt * 0.65;

    // 法向碰撞 + 穿透偏移冲量
    let v_contact = *vel + ang.cross(r);
    let vn = v_contact.dot(normal);
    let r_cross_n = r.cross(normal);
    let denom_n = inv_mass + (r_cross_n * inv_inertia).cross(r).dot(normal);
    if denom_n > 1e-5 {
        let impact_j = if vn < -0.02 { -vn / denom_n } else { 0.0 };
        let bias_j = (contact.depth * 0.8 / denom_n).clamp(0.0, 0.35);
        let j = (impact_j + bias_j).clamp(0.0, 1.2);
        *vel += normal * (impact_j.clamp(0.0, 1.0) + bias_j * 0.15);
        *ang += r.cross(normal * j) * inv_inertia;
    }

    // 切向摩擦冲量
    let v_contact_after = *vel + ang.cross(r);
    let tangent_v = v_contact_after - normal * v_contact_after.dot(normal);
    let tangent_len = tangent_v.length();
    if tangent_len > 1e-4 {
        let tangent = tangent_v / tangent_len;
        let r_cross_t = r.cross(tangent);
        let denom = inv_mass + (r_cross_t * inv_inertia).cross(r).dot(tangent);
        if denom > 1e-5 {
            let max_friction = 0.35;
            let j = (-tangent_len / denom).clamp(-max_friction, max_friction);
            *vel += tangent * j;
            *ang += r.cross(tangent * j) * inv_inertia;
        }
    }

    if vel.y < 0.0 {
        vel.y = 0.0;
    }
    *vel = vel.clamp_length_max(6.0);
    *ang = ang.clamp_length_max(8.0);
}

/// 逐帧物理推进：重力、积分、碰撞检测与响应（地面 + 物体间双体碰撞）
pub fn apply_physics(
    instances: &mut [math::InstanceData],
    instance_physics: &mut Vec<InstancePhysics>,
    model_colliders: &HashMap<u32, ModelCollider>,
    editing: bool,
    delta_time: f32,
) {
    // #region debug-point window-freeze-physics
    let physics_t0 = std::time::Instant::now();
    let active_physics = instance_physics.iter().filter(|p| p.gravity_enabled).count();
    // #endregion debug-point window-freeze-physics
    sync_instance_physics_len(instances, instance_physics);
    let frame_dt = delta_time.min(1.0 / 30.0);

    // ----------------------------------------------------------------
    // 阶段 1: 重力与积分（预更新所有物体的位置）
    // ----------------------------------------------------------------
    for idx in 0..instances.len() {
        if !instance_physics[idx].gravity_enabled || editing {
            continue;
        }
        if instance_physics[idx].sleeping {
            instance_physics[idx].sleeping = false;
        }
        instance_physics[idx].velocity.y -= 9.8 * frame_dt;

        let mut mat = glam::Mat4::from_cols_array_2d(&instances[idx].model_matrix);
        let (scale, mut rot, mut pos) = mat.to_scale_rotation_translation();

        pos += instance_physics[idx].velocity * frame_dt;

        let angular_speed = instance_physics[idx].angular_velocity.length();
        if angular_speed > 1e-5 {
            let axis = instance_physics[idx].angular_velocity / angular_speed;
            rot = glam::Quat::from_axis_angle(axis, angular_speed * frame_dt) * rot;
        }

        instance_physics[idx].velocity = (instance_physics[idx].velocity * 0.99).clamp_length_max(8.0);
        instance_physics[idx].angular_velocity = (instance_physics[idx].angular_velocity * 0.98).clamp_length_max(12.0);

        mat = glam::Mat4::from_scale_rotation_translation(scale, rot.normalize(), pos);
        instances[idx].model_matrix = mat.to_cols_array_2d();
    }

    // ----------------------------------------------------------------
    // 阶段 2: 包围球轻量唤醒
    // ----------------------------------------------------------------
    for idx in 0..instances.len() {
        if !instance_physics[idx].gravity_enabled || instance_physics[idx].sleeping {
            continue;
        }
        let Some(collider_a) = model_colliders.get(&instances[idx].model_id) else { continue; };
        let mat_a = glam::Mat4::from_cols_array_2d(&instances[idx].model_matrix);
        let (scale_a, _, center_a) = mat_a.to_scale_rotation_translation();
        let radius_a = (collider_a.bounds_max - collider_a.bounds_min).length() * scale_a.abs().max_element() * 0.5;
        for other_idx in 0..instances.len() {
            if other_idx == idx || !instance_physics[other_idx].sleeping {
                continue;
            }
            let Some(collider_b) = model_colliders.get(&instances[other_idx].model_id) else { continue; };
            let mat_b = glam::Mat4::from_cols_array_2d(&instances[other_idx].model_matrix);
            let (scale_b, _, center_b) = mat_b.to_scale_rotation_translation();
            let radius_b = (collider_b.bounds_max - collider_b.bounds_min).length() * scale_b.abs().max_element() * 0.5;
            if center_a.distance(center_b) < radius_a + radius_b + 0.05 {
                instance_physics[other_idx].sleeping = false;
            }
        }
    }

    // ----------------------------------------------------------------
    // 阶段 3: 统一碰撞解决（松弛迭代）
    // ----------------------------------------------------------------
    for _relaxation in 0..2 {
        for idx in 0..instances.len() {
            if !instance_physics[idx].gravity_enabled || editing {
                continue;
            }
            if instance_physics[idx].sleeping {
                continue;
            }

            // ---- 3a. 地面碰撞解算 ----
            let ground_contacts = detect_ground_collisions(instances, model_colliders, idx);
            if !ground_contacts.is_empty() {
                let mut mat_a = glam::Mat4::from_cols_array_2d(&instances[idx].model_matrix);
                let (scale_a, mut rot_a, mut pos_a) = mat_a.to_scale_rotation_translation();
                let mut vel_a = instance_physics[idx].velocity;
                let mut ang_a = instance_physics[idx].angular_velocity;

                let max_depth = ground_contacts
                    .iter()
                    .map(|contact| contact.depth)
                    .fold(0.0, f32::max);
                pos_a.y += max_depth + 0.001;

                for contact in ground_contacts.iter().copied().take(4) {
                    resolve_ground_contact_with_rotation(pos_a, &mut vel_a, &mut ang_a, contact, frame_dt);
                }
                let up = rot_a * glam::Vec3::Y;
                let upright_axis = up.cross(glam::Vec3::Y);
                if upright_axis.length_squared() > 1e-5 {
                    ang_a += upright_axis.clamp_length_max(0.35);
                }
                vel_a.x *= 0.9;
                vel_a.z *= 0.9;
                ang_a *= 0.94;

                if vel_a.length() < 0.03 && ang_a.length() < 0.03 && max_depth < 0.003 {
                    vel_a = glam::Vec3::ZERO;
                    ang_a = glam::Vec3::ZERO;
                    instance_physics[idx].sleeping = true;
                }
                instance_physics[idx].velocity = vel_a.clamp_length_max(8.0);
                instance_physics[idx].angular_velocity = ang_a.clamp_length_max(12.0);
                mat_a = glam::Mat4::from_scale_rotation_translation(scale_a, rot_a.normalize(), pos_a);
                instances[idx].model_matrix = mat_a.to_cols_array_2d();
            }

            // ---- 3b. 模型间对称式双体碰撞解算 ----
            let mesh_contacts = detect_mesh_collisions_symmetric_detailed(instances, model_colliders, idx);
            let mut resolved_pairs = 0;
            for m_contact in mesh_contacts {
                if resolved_pairs >= 3 {
                    break;
                }
                let other_idx = m_contact.other_idx;

                let mut mat_a = glam::Mat4::from_cols_array_2d(&instances[idx].model_matrix);
                let (scale_a, mut rot_a, mut pos_a) = mat_a.to_scale_rotation_translation();
                let mut vel_a = instance_physics[idx].velocity;
                let mut ang_a = instance_physics[idx].angular_velocity;

                let mut mat_b = glam::Mat4::from_cols_array_2d(&instances[other_idx].model_matrix);
                let (scale_b, mut rot_b, mut pos_b) = mat_b.to_scale_rotation_translation();
                let mut vel_b = instance_physics[other_idx].velocity;
                let mut ang_b = instance_physics[other_idx].angular_velocity;

                let has_physics_b = instance_physics[other_idx].gravity_enabled;
                if has_physics_b && !instance_physics[other_idx].sleeping && other_idx < idx {
                    continue;
                }

                resolve_rigid_body_contact_two_bodies(
                    &mut pos_a, &mut vel_a, &mut ang_a, true,
                    &mut pos_b, &mut vel_b, &mut ang_b, has_physics_b,
                    m_contact.contact,
                    0.1,
                    0.3,
                );

                resolved_pairs += 1;

                instance_physics[idx].velocity = vel_a.clamp_length_max(8.0);
                instance_physics[idx].angular_velocity = ang_a.clamp_length_max(12.0);
                instance_physics[idx].sleeping = false;
                mat_a = glam::Mat4::from_scale_rotation_translation(scale_a, rot_a.normalize(), pos_a);
                instances[idx].model_matrix = mat_a.to_cols_array_2d();

                if has_physics_b {
                    instance_physics[other_idx].velocity = vel_b.clamp_length_max(8.0);
                    instance_physics[other_idx].angular_velocity = ang_b.clamp_length_max(12.0);
                    instance_physics[other_idx].sleeping = false;
                    mat_b = glam::Mat4::from_scale_rotation_translation(scale_b, rot_b.normalize(), pos_b);
                    instances[other_idx].model_matrix = mat_b.to_cols_array_2d();
                }
            }
        }
    }

    // #region debug-point window-freeze-physics
    let physics_ms = physics_t0.elapsed().as_secs_f64() * 1000.0;
    if physics_ms > 25.0 {
        eprintln!("[debug-window-freeze] apply_physics active={} instances={} elapsed_ms={:.2}", active_physics, instances.len(), physics_ms);
    }
    // #endregion debug-point window-freeze-physics
}
