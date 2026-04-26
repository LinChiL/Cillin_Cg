use glam::{Vec3, Mat4, Vec4Swizzles};
use crate::scene::Entity;
use crate::undo::{UndoManager, EntityAction};

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum EditMode {
    Idle,
    Grab,
    Rotate,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    None,
    X,
    Y,
    Z,
}

pub struct EditorState {
    pub selected_idx: Option<usize>,
    pub active_model_id: u32,
    pub edit_mode: EditMode,
    pub selected_axis: Axis,
    pub undo_manager: UndoManager,
    pub edit_start_state: Option<EntityAction>,
    pub is_ser_spawn_mode: bool,
    pub ser_spawn_model_id: u32,
}

impl EditorState {
    pub fn new() -> Self {
        Self {
            selected_idx: None,
            active_model_id: 1,
            edit_mode: EditMode::Idle,
            selected_axis: Axis::None,
            undo_manager: UndoManager::new(50),
            edit_start_state: None,
            is_ser_spawn_mode: false,
            ser_spawn_model_id: 0,
        }
    }

    pub fn select_entity(&mut self, idx: Option<usize>) {
        self.selected_idx = idx;
        if idx.is_none() {
            self.edit_mode = EditMode::Idle;
            self.selected_axis = Axis::None;
        }
    }

    pub fn begin_edit(&mut self, mode: EditMode, entities: &[Entity]) {
        if let Some(idx) = self.selected_idx {
            if idx < entities.len() {
                self.edit_mode = mode;
                self.edit_start_state = Some(EntityAction {
                    entity_id: idx,
                    old_position: entities[idx].position,
                    old_rotation: entities[idx].rotation,
                    old_scale: entities[idx].scale,
                    new_position: entities[idx].position,
                    new_rotation: entities[idx].rotation,
                    new_scale: entities[idx].scale,
                });
            }
        }
    }

    pub fn end_edit(&mut self, entities: &mut [Entity]) {
        if let Some(mut action) = self.edit_start_state.take() {
            if let Some(idx) = self.selected_idx {
                action.new_position = entities[idx].position;
                action.new_rotation = entities[idx].rotation;
                action.new_scale = entities[idx].scale;
                self.undo_manager.push(action);
            }
        }
        self.edit_mode = EditMode::Idle;
        self.selected_axis = Axis::None;
    }

    pub fn undo(&mut self, entities: &mut Vec<Entity>) {
        self.undo_manager.undo(entities);
    }

    pub fn redo(&mut self, entities: &mut Vec<Entity>) {
        self.undo_manager.redo(entities);
    }

    pub fn apply_grab(&mut self, entities: &mut [Entity], dx: f32, dy: f32) {
        if let Some(idx) = self.selected_idx {
            match self.selected_axis {
                Axis::X => entities[idx].position.x += dx,
                Axis::Y => entities[idx].position.y += dy,
                Axis::Z => entities[idx].position.z += dy,
                Axis::None => {
                    entities[idx].position.x += dx;
                    entities[idx].position.z += dy;
                }
            }
        }
    }

    pub fn apply_scale(&mut self, entities: &mut [Entity], scale_delta: f32) {
        if let Some(idx) = self.selected_idx {
            let scale_factor = 1.0 + scale_delta;
            entities[idx].scale *= scale_factor;
        }
    }

    pub fn apply_rotate(&mut self, entities: &mut [Entity], dx: f32, dy: f32) {
        if let Some(idx) = self.selected_idx {
            entities[idx].rotation.y += dx;
            entities[idx].rotation.x += dy;
        }
    }

    pub fn enable_ser_spawn_mode(&mut self, model_id: u32) {
        self.is_ser_spawn_mode = true;
        self.ser_spawn_model_id = model_id;
    }

    pub fn disable_ser_spawn_mode(&mut self) {
        self.is_ser_spawn_mode = false;
        self.ser_spawn_model_id = 0;
    }

    pub fn handle_key_press(&mut self, c: char, entities: &mut Vec<Entity>) -> Option<EditorCommand> {
        match c {
            'g' | 'G' => {
                if self.selected_idx.is_some() {
                    self.begin_edit(EditMode::Grab, entities);
                    Some(EditorCommand::StartGrab)
                } else {
                    None
                }
            }
            's' | 'S' => {
                if self.selected_idx.is_some() {
                    self.begin_edit(EditMode::Scale, entities);
                    Some(EditorCommand::StartScale)
                } else {
                    None
                }
            }
            'r' | 'R' => {
                if self.selected_idx.is_some() {
                    self.begin_edit(EditMode::Rotate, entities);
                    Some(EditorCommand::StartRotate)
                } else {
                    None
                }
            }
            'x' | 'X' => {
                if self.edit_mode == EditMode::Grab {
                    self.selected_axis = Axis::X;
                    Some(EditorCommand::SelectAxis(Axis::X))
                } else {
                    None
                }
            }
            'y' | 'Y' => {
                if self.edit_mode == EditMode::Grab {
                    self.selected_axis = Axis::Y;
                    Some(EditorCommand::SelectAxis(Axis::Y))
                } else {
                    self.redo(entities);
                    Some(EditorCommand::Redo)
                }
            }
            'z' | 'Z' => {
                if self.edit_mode == EditMode::Grab {
                    self.selected_axis = Axis::Z;
                    Some(EditorCommand::SelectAxis(Axis::Z))
                } else {
                    None
                }
            }
            'u' | 'U' => {
                self.undo(entities);
                Some(EditorCommand::Undo)
            }
            _ => None,
        }
    }
}

pub enum EditorCommand {
    StartGrab,
    StartScale,
    StartRotate,
    SelectAxis(Axis),
    Undo,
    Redo,
}

pub fn get_clicked_entity<'a>(
    entities: &'a [Entity],
    camera_eye: Vec3,
    ray_dir: Vec3,
) -> Option<&'a Entity> {
    let mut closest_entity: Option<&'a Entity> = None;
    let mut closest_t = f32::MAX;

    for entity in entities {
        let model_matrix = Mat4::from_cols_array_2d(&entity.get_model_matrix());
        let entity_center = model_matrix.transform_point3(Vec3::ZERO);
        let entity_scale = entity.scale.max_element();
        
        let offset = entity_center - camera_eye;
        let b = offset.dot(ray_dir);
        let c = offset.dot(offset) - entity_scale * entity_scale * 0.5;
        
        let discriminant = b * b - c;
        if discriminant < 0.0 {
            continue;
        }
        
        let t = b - discriminant.sqrt();
        if t > 0.0 && t < closest_t {
            closest_t = t;
            closest_entity = Some(entity);
        }
    }

    closest_entity
}

pub fn get_ray_from_screen(
    mouse_pos: (f64, f64),
    screen_width: u32,
    screen_height: u32,
    view: Mat4,
    proj: Mat4,
    camera_eye: Vec3,
) -> Vec3 {
    let ndc = glam::vec4(
        (mouse_pos.0 as f32 / screen_width as f32) * 2.0 - 1.0,
        1.0 - (mouse_pos.1 as f32 / screen_height as f32) * 2.0,
        0.0, 1.0
    );
    let inv_vp = (proj * view).inverse();
    let world_pos = inv_vp * ndc;
    let world_pos = world_pos.xyz() / world_pos.w;
    (world_pos - camera_eye).normalize()
}

pub fn get_click_ground_pos(
    mouse_pos: (f64, f64),
    screen_width: u32,
    screen_height: u32,
    view: Mat4,
    proj: Mat4,
    camera_eye: Vec3,
) -> Vec3 {
    let ray_dir = get_ray_from_screen(mouse_pos, screen_width, screen_height, view, proj, camera_eye);
    
    if ray_dir.y.abs() < 0.001 {
        return camera_eye + ray_dir * 10.0;
    }
    
    let t = -camera_eye.y / ray_dir.y;
    if t < 0.0 {
        return camera_eye + ray_dir * 10.0;
    }
    camera_eye + ray_dir * t
}