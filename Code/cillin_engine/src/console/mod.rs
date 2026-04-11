use wgpu::{Device, Queue, TextureView, CommandEncoder};
use std::collections::VecDeque;

pub struct Console {
    pub is_open: bool,
    pub buffer: String,
    pub history: VecDeque<String>,
    pub history_index: Option<usize>,
    pub visible_lines: usize,
}

impl Console {
    pub fn new() -> Self {
        Self {
            is_open: false,
            buffer: String::new(),
            history: VecDeque::with_capacity(50),
            history_index: None,
            visible_lines: 10,
        }
    }

    pub fn add_char(&mut self, c: char) {
        self.buffer.push(c);
        println!("> {}", self.buffer);
    }

    pub fn remove_char(&mut self) {
        if self.buffer.pop().is_some() {
            println!("> {}", self.buffer);
        }
    }

    pub fn execute(&mut self) -> String {
        let command = self.buffer.trim().to_string();
        if !command.is_empty() {
            self.history.push_front(command.clone());
            if self.history.len() > 50 {
                self.history.pop_back();
            }
        }
        self.buffer.clear();
        self.history_index = None;
        command
    }

    pub fn navigate_history(&mut self, direction: i32) {
        let len = self.history.len();
        if len == 0 {
            return;
        }

        self.history_index = match self.history_index {
            None => {
                if direction > 0 {
                    Some(0)
                } else {
                    None
                }
            }
            Some(index) => {
                let new_index = (index as i32 + direction).clamp(0, len as i32 - 1) as usize;
                Some(new_index)
            }
        };

        if let Some(index) = self.history_index {
            self.buffer = self.history[index].clone();
        } else {
            self.buffer.clear();
        }
    }

    pub fn render(&self, _device: &Device, _queue: &Queue, _view: &TextureView, _encoder: &mut CommandEncoder) {
        if !self.is_open {
            return;
        }

        // 这里将来可以用实际的文本渲染库实现
        // 暂时只是占位
    }
}
