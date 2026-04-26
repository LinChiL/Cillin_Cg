use std::sync::Arc;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::WindowBuilder,
};

use cillin_engine::core::CoreState;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("Developer Mode Engine").build(&event_loop).unwrap());

    let mut state: CoreState<'_> = pollster::block_on(CoreState::new(window.clone()));

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { ref event, window_id } if window_id == state.window.id() => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(s) => state.resize(*s),
            WindowEvent::MouseInput { button, state: button_state, .. } => {
                state.input_state.process_event(event);
                if state.console.is_open { return; }

                if button == &MouseButton::Middle {
                    if *button_state == ElementState::Pressed {
                        state.input_state.last_mouse_pos = Some(state.input_state.mouse_pos);
                    } else {
                        state.input_state.last_mouse_pos = None;
                    }
                } else if button == &MouseButton::Left {
                    if *button_state == ElementState::Pressed {
                        if state.editor.is_ser_spawn_mode {
                            return;
                        }
                        if let Some(entity) = state.get_clicked_entity() {
                            println!("{}", entity.model_id);
                            println!("{}", entity.code);
                            state.editor.select_entity(Some(state.entities.iter().position(|e| e.id == entity.id).unwrap()));
                        } else if state.editor.selected_idx.is_some() {
                            println!("cancel");
                            state.editor.select_entity(None);
                        }
                    } else if state.editor.edit_mode != cillin_engine::editor::EditMode::Idle {
                        state.editor.end_edit(&mut state.entities);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x, position.y);
                if let Some(last_pos) = state.input_state.last_mouse_pos {
                    let dx = -(new_pos.0 - last_pos.0) as f32;
                    let dy = (new_pos.1 - last_pos.1) as f32;
                    state.camera.yaw -= dx * 0.001;
                    state.camera.pitch -= dy * 0.001;
                    state.camera.pitch = state.camera.pitch.max(-std::f32::consts::FRAC_PI_2).min(std::f32::consts::FRAC_PI_2);
                } else if state.editor.selected_idx.is_some() && state.editor.edit_mode != cillin_engine::editor::EditMode::Idle {
                    match state.editor.edit_mode {
                        cillin_engine::editor::EditMode::Grab => {
                            let dx = (new_pos.0 - state.input_state.mouse_pos.0) as f32 * 0.01;
                            let dy = (state.input_state.mouse_pos.1 - new_pos.1) as f32 * 0.01;
                            state.editor.apply_grab(&mut state.entities, dx, dy);
                        }
                        cillin_engine::editor::EditMode::Scale => {
                            let dy = (new_pos.1 - state.input_state.mouse_pos.1) as f32 * 0.005;
                            state.editor.apply_scale(&mut state.entities, dy);
                        }
                        cillin_engine::editor::EditMode::Rotate => {
                            let dx = (new_pos.0 - state.input_state.mouse_pos.0) as f32 * 0.01;
                            let dy = (new_pos.1 - state.input_state.mouse_pos.1) as f32 * 0.01;
                            state.editor.apply_rotate(&mut state.entities, dx, dy);
                        }
                        cillin_engine::editor::EditMode::Idle => {}
                    }
                };
                state.input_state.last_mouse_pos = if state.input_state.is_middle_mouse_pressed {
                    Some(new_pos)
                } else {
                    None
                };
                state.input_state.mouse_pos = new_pos;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if state.console.is_open {
                    if event.state == ElementState::Pressed {
                        if matches!(event.logical_key, winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace)) {
                            state.console.remove_char();
                        } else if let Some(text) = &event.text {
                            if text == "\r" {
                                state.execute_command();
                            } else if text != "\x08" {
                                state.console.add_char(text.chars().next().unwrap());
                            }
                        } else if matches!(event.logical_key, winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape)) {
                            state.console.is_open = false;
                        }
                    }
                    return;
                }

                state.input_state.process_key_event(event);

                if let winit::keyboard::Key::Character(c) = &event.logical_key {
                    if event.state == ElementState::Pressed {
                        match c.as_str() {
                            "p" => state.save_scene(),
                            "/" => state.console.is_open = true,
                            "g" | "G" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            "s" | "S" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            "r" | "R" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            "x" | "X" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            "y" | "Y" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            "z" | "Z" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            "u" | "U" => {
                                state.editor.handle_key_press(c.chars().next().unwrap(), &mut state.entities);
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta: _, .. } => {
                state.input_state.process_event(event);
                state.camera.eye += state.camera.get_forward() * state.input_state.scroll_delta * 0.1;
                state.input_state.reset_scroll();
            }
            _ => {}
        }
        Event::AboutToWait => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.render_context.size),
                Err(e) => eprintln!("{:?}", e),
            }
            state.window.request_redraw();
        }
        _ => {}
    }).unwrap();
}