use std::collections::VecDeque;
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct EntityAction {
    pub entity_id: usize,
    pub old_position: Vec3,
    pub old_rotation: Vec3,
    pub old_scale: Vec3,
    pub new_position: Vec3,
    pub new_rotation: Vec3,
    pub new_scale: Vec3,
}

pub struct UndoManager {
    undo_stack: VecDeque<EntityAction>,
    redo_stack: VecDeque<EntityAction>,
    max_history: usize,
}

impl UndoManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            undo_stack: VecDeque::with_capacity(max_history),
            redo_stack: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    pub fn push(&mut self, action: EntityAction) {
        if self.undo_stack.len() >= self.max_history {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(action);
        self.redo_stack.clear();
    }

    pub fn undo(&mut self, entities: &mut Vec<crate::scene::Entity>) {
        if let Some(action) = self.undo_stack.pop_back() {
            if let Some(entity) = entities.get_mut(action.entity_id) {
                entity.position = action.old_position;
                entity.rotation = action.old_rotation;
                entity.scale = action.old_scale;
            }
            self.redo_stack.push_back(action);
        }
    }

    pub fn redo(&mut self, entities: &mut Vec<crate::scene::Entity>) {
        if let Some(action) = self.redo_stack.pop_back() {
            if let Some(entity) = entities.get_mut(action.entity_id) {
                entity.position = action.new_position;
                entity.rotation = action.new_rotation;
                entity.scale = action.new_scale;
            }
            self.undo_stack.push_back(action);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}
