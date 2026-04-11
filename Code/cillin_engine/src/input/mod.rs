use winit::event::{WindowEvent, ElementState, MouseButton, MouseScrollDelta, KeyEvent};

#[derive(Debug, Default)]
pub struct InputState {
    pub is_w_pressed: bool,
    pub is_s_pressed: bool,
    pub is_a_pressed: bool,
    pub is_d_pressed: bool,
    pub is_q_pressed: bool,
    pub is_e_pressed: bool,
    pub mouse_pos: (f64, f64),
    pub is_middle_mouse_pressed: bool,
    pub last_mouse_pos: Option<(f64, f64)>,
    pub scroll_delta: f32,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event: input, .. } => {
                self.process_key_event(input);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if button == &MouseButton::Middle {
                    self.is_middle_mouse_pressed = state == &ElementState::Pressed;
                    if self.is_middle_mouse_pressed {
                        self.last_mouse_pos = Some(self.mouse_pos);
                    } else {
                        self.last_mouse_pos = None;
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.scroll_delta += *y;
                    }
                    MouseScrollDelta::PixelDelta(_) => {}
                }
            }
            _ => {}
        }
    }

    pub fn process_key_event(&mut self, input: &KeyEvent) {
        let pressed = input.state == ElementState::Pressed;
        match &input.logical_key {
            winit::keyboard::Key::Character(c) => match c.as_str() {
                "w" => self.is_w_pressed = pressed,
                "s" => self.is_s_pressed = pressed,
                "a" => self.is_a_pressed = pressed,
                "d" => self.is_d_pressed = pressed,
                "q" => self.is_q_pressed = pressed,
                "e" => self.is_e_pressed = pressed,
                _ => {}
            },
            _ => {}
        }
    }

    pub fn reset_scroll(&mut self) {
        self.scroll_delta = 0.0;
    }
}
