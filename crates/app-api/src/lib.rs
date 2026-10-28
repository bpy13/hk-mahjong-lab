//! `mahjong-app-api`: the app-facing adapter over the Hong Kong mahjong engine.
//!
//! Implements the calls of hk-games `docs/engine-interface.md` (v0) as JSON
//! strings in and out, so the same boundary serves WebAssembly, native and the
//! server. [`Api`] is the pure-Rust core; the `wasm32` build adds a thin
//! `wasm-bindgen` class ([`wasm::MahjongApi`]) with the same calls.
//!
//! The shapes, keys and numbers are documented in `README.md`.

pub mod api;
pub mod reasons;
pub mod shapes;
pub mod solver;

pub use api::{Api, BOT_LEVELS};
pub use shapes::API_VERSION;

/// The WebAssembly surface: one class, every call a method taking and
/// returning JSON strings.
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct MahjongApi {
        inner: crate::Api,
    }

    #[wasm_bindgen]
    impl MahjongApi {
        #[wasm_bindgen(constructor)]
        pub fn new() -> MahjongApi {
            MahjongApi {
                inner: crate::Api::new(),
            }
        }

        pub fn meta(&self) -> String {
            self.inner.meta()
        }

        pub fn new_game(&mut self, config: &str) -> String {
            self.inner.new_game(config)
        }

        pub fn view(&self, game_id: &str, seat: usize) -> String {
            self.inner.view(game_id, seat)
        }

        pub fn legal_actions(&self, game_id: &str) -> String {
            self.inner.legal_actions(game_id)
        }

        pub fn apply(&mut self, game_id: &str, action: &str) -> String {
            self.inner.apply(game_id, action)
        }

        pub fn bot_action(&self, game_id: &str, level: &str, budget_ms: u32) -> String {
            self.inner.bot_action(game_id, level, budget_ms)
        }

        pub fn hint(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
            self.inner.hint(game_id, seat, budget_ms)
        }

        pub fn hint_with_samples(
            &self,
            game_id: &str,
            seat: usize,
            budget_ms: u32,
            samples: u32,
        ) -> String {
            // `u32`, so JavaScript passes a plain number rather than a BigInt.
            self.inner
                .hint_with_samples(game_id, seat, budget_ms, samples as usize)
        }

        pub fn review(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
            self.inner.review(game_id, seat, budget_ms)
        }

        pub fn grade(&self, position: &str, action: &str, budget_ms: u32) -> String {
            self.inner.grade(position, action, budget_ms)
        }

        pub fn snapshot(&self, game_id: &str) -> String {
            self.inner.snapshot(game_id)
        }

        pub fn restore(&mut self, position: &str) -> String {
            self.inner.restore(position)
        }

        pub fn drop_game(&mut self, game_id: &str) -> String {
            self.inner.drop_game(game_id)
        }
    }

    impl Default for MahjongApi {
        fn default() -> Self {
            MahjongApi::new()
        }
    }
}
