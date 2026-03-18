//! Backend-neutral interactive state and input mapping for `lbm-live-viewer`.
//!
//! The current frontend uses slice controls because the first backend adapter
//! exposes real 3D volume data. Later 3D camera-capable backends can extend the
//! same state container with orbit or fly-camera parameters without changing the
//! window loop.

use gororoba_view_core::GridShape3d;
pub use gororoba_view_raster::SliceAxis;
use minifb::{Key, KeyRepeat, Window};

/// Mutable viewer interaction state shared by the frontend loop and transport.
#[derive(Debug, Clone)]
pub struct ViewerInteractionState {
    pub paused: bool,
    pub steps_per_frame: usize,
    pub slice_axis: SliceAxis,
    pub slice_index: usize,
}

impl ViewerInteractionState {
    /// Initialize interaction state for a grid and default step cadence.
    #[must_use]
    pub fn new(grid: GridShape3d, steps_per_frame: usize) -> Self {
        Self {
            paused: false,
            steps_per_frame,
            slice_axis: SliceAxis::Z,
            slice_index: grid.nz as usize / 2,
        }
    }

    /// Clamp the currently selected slice index to the chosen axis.
    pub fn clamp_slice_index(&mut self, grid: GridShape3d) {
        self.slice_index = self
            .slice_index
            .min(slice_axis_max_index(self.slice_axis, grid));
    }
}

/// One-frame actions emitted from the input mapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct ViewerActions {
    pub request_reset: bool,
}

/// Apply keyboard input to the current interaction state.
#[must_use]
pub fn apply_window_input(
    window: &Window,
    state: &mut ViewerInteractionState,
    grid: GridShape3d,
) -> ViewerActions {
    let mut actions = ViewerActions::default();

    if window.is_key_pressed(Key::Space, KeyRepeat::No) {
        state.paused = !state.paused;
    }
    if window.is_key_pressed(Key::R, KeyRepeat::No) {
        actions.request_reset = true;
    }
    if window.is_key_pressed(Key::Equal, KeyRepeat::Yes)
        || window.is_key_pressed(Key::NumPadPlus, KeyRepeat::Yes)
    {
        state.steps_per_frame = (state.steps_per_frame + 1).min(100);
    }
    if window.is_key_pressed(Key::Minus, KeyRepeat::Yes)
        || window.is_key_pressed(Key::NumPadMinus, KeyRepeat::Yes)
    {
        state.steps_per_frame = state.steps_per_frame.saturating_sub(1).max(1);
    }
    if window.is_key_pressed(Key::Right, KeyRepeat::No) {
        state.slice_axis = next_slice_axis(state.slice_axis);
        state.clamp_slice_index(grid);
    }
    if window.is_key_pressed(Key::Left, KeyRepeat::No) {
        state.slice_axis = previous_slice_axis(state.slice_axis);
        state.clamp_slice_index(grid);
    }
    if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
        state.slice_index =
            (state.slice_index + 1).min(slice_axis_max_index(state.slice_axis, grid));
    }
    if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
        state.slice_index = state.slice_index.saturating_sub(1);
    }

    actions
}

#[must_use]
pub fn next_slice_axis(axis: SliceAxis) -> SliceAxis {
    match axis {
        SliceAxis::X => SliceAxis::Y,
        SliceAxis::Y => SliceAxis::Z,
        SliceAxis::Z => SliceAxis::X,
    }
}

#[must_use]
pub fn previous_slice_axis(axis: SliceAxis) -> SliceAxis {
    match axis {
        SliceAxis::X => SliceAxis::Z,
        SliceAxis::Y => SliceAxis::X,
        SliceAxis::Z => SliceAxis::Y,
    }
}

#[must_use]
pub fn slice_axis_max_index(axis: SliceAxis, grid: GridShape3d) -> usize {
    axis.max_index(grid)
}

#[must_use]
pub fn slice_axis_label(axis: SliceAxis) -> &'static str {
    match axis {
        SliceAxis::X => "X",
        SliceAxis::Y => "Y",
        SliceAxis::Z => "Z",
    }
}
