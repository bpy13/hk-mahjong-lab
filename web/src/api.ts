// Types and calls for the FastAPI server (python/mahjong/server/app.py).

export interface Meld {
  kind: 'chow' | 'pung' | 'kong'
  tile: number
  tiles: number[]
  concealed: boolean
  from: number | null
}

export interface Observation {
  seat: number
  phase: 'turn' | 'claim' | 'rob' | 'over'
  current_player: number | null
  turn: number
  wall_remaining: number
  hand: number[]
  last_drawn: number | null
  melds: Meld[][]
  bonus: number[][]
  concealed_counts: number[]
  river: [number, number, boolean][]
  claim: [number, number] | null
  liable: (number | null)[]
  legal: number[]
  unseen: number[]
}

export interface Score {
  faan: number
  total: number
  limit: boolean
  patterns: [number, number][]
}

export interface WinRecord {
  seat: number
  score: Score
  self_drawn: boolean
  from: number | null
  win_tile: number | null
  paid_by: number[]
  liable: number | null
  robbed: boolean
}

export interface HandResult {
  wins: WinRecord[]
  deltas: number[]
}

export interface Rules {
  flowers: boolean
  min_faan: number
  faan_cap: number
  payout: string
  multiple_winners: string
  twelve_tile_liability: boolean
  prevailing_wind: number
}

export interface GameView {
  id: string
  seed: number
  level: string
  human: number
  rules: Rules
  observation: Observation
  legal: { action: number; label: string }[]
  bot_to_act: boolean
  events: Record<string, unknown>[]
  result: HandResult | null
  hands?: number[][]
}

export interface MoveStats {
  action: number
  text: string
  label: string
  n: number
  mean: number
  ci95: number
  win_rate: number
  deal_in_rate: number
  draw_rate: number
}

export interface Target {
  plan: string
  structure: string
  pungs: number[]
  faan: number
  limit: boolean
}

export interface DiscardRow {
  tile: number
  shanten: number
  ukeire: number
  faan_shanten: number
  faan_ukeire: number
  target: Target | null
  solo: { win_rate: number; mean_faan: number; mean_draws: number } | null
}

export interface Hints {
  solver: { seat: number; samples: number; elapsed_ms: number; moves: MoveStats[] }
  danger: { tiles: number[]; threat: number[] }
  progress: { shanten: number; faan_shanten: number; target: Target | null }
  discards?: DiscardRow[]
}

export interface ReviewItem {
  index: number
  phase: string
  observation: Observation
  chosen: number
  chosen_label: string
  best: MoveStats
  chosen_stats: MoveStats | null
  loss: number | null
}

export interface Meta {
  levels: Record<string, { label: string; spec: string; about: string }>
  patterns: Record<string, string>
  tiles: string[]
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return (await res.json()) as T
}

export const api = {
  meta: () => call<Meta>('GET', '/api/meta'),
  newGame: (level: string, seat: number | null, rules: Rules) =>
    call<GameView>('POST', '/api/games', { level, seat, rules }),
  act: (id: string, action: number) => call<GameView>('POST', `/api/games/${id}/act`, { action }),
  step: (id: string) => call<GameView>('POST', `/api/games/${id}/step`),
  hints: (id: string, ms: number) => call<Hints>('GET', `/api/games/${id}/hints?ms=${ms}`),
  review: (id: string) => call<{ decisions: ReviewItem[] }>('GET', `/api/games/${id}/review`),
}

export const DEFAULT_RULES: Rules = {
  flowers: true,
  min_faan: 3,
  faan_cap: 10,
  payout: 'half_spicy',
  multiple_winners: 'all',
  twelve_tile_liability: true,
  prevailing_wind: 0,
}
